# zfg 第四部分（模型七）：
# 采用原始的raw eeg 20个电极的相邻电极脑电波数据做差，通过巴特沃斯带通滤波器低通滤波处理得到（10000，18）即为50s的18个特征的脑电波数据。
'''
模型七：官方的的raw eeg 数据，训练EEGNet模型
'''
import gc
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import scipy.signal as scisig
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lion_pytorch import Lion
from matplotlib import pyplot as plt
from ranger21 import Ranger21
from scipy.signal import butter, lfilter
from sklearn.model_selection import GroupKFold
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    OneCycleLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from sam.sam import SAM

# 如果你使用自定义的 Lion 优化器，需要确保它被正确导入
sys.path.append('../kaggle_kl_div')
from kaggle_kl_div import kaggle_kl_div

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# "PRETTY_NAME=\"\K([^\"]*)"
# print(f"BUILD_DATE={os.environ['BUILD_DATE']}, CONTAINER_NAME={os.environ['CONTAINER_NAME']}")

try:
    print(
        f"PyTorch Version:{torch.__version__}, CUDA is available:{torch.cuda.is_available()}, Version CUDA:{torch.version.cuda}"
    )
    print(
        f"Device Capability:{torch.cuda.get_device_capability()}, {torch.cuda.get_arch_list()}"
    )
    print(
        f"CuDNN Enabled:{torch.backends.cudnn.enabled}, Version:{torch.backends.cudnn.version()}"
    )
except Exception:
    pass

OUTPUT_DIR = "../hms-harmful-brain-activity-classification/output_7/"


class CFG:
    VERSION = '1'

    wandb = False
    debug = False
    create_eegs = True
    apex = True
    visualize = False
    save_all_models = True

    if debug:
        num_workers = 0
        parallel = False
    else:
        num_workers = os.cpu_count()
        parallel = True

    model_name = "resnet1d_gru"
    # optimizer = "Adan"
    optimizer = "AdamW"

    factor = 0.9
    eps = 1e-6
    lr = 8e-3
    min_lr = 1e-6

    batch_size = 64
    batch_koef_valid = 2
    batch_scheduler = True
    weight_decay = 1e-2
    gradient_accumulation_steps = 1
    max_grad_norm = 1e7

    fixed_kernel_size = 5
    # linear_layer_features = 424
    # kernels = [3, 5, 7, 9]
    # linear_layer_features = 448  # Full Signal = 10_000
    # linear_layer_features = 352  # Half Signal = 5_000
    linear_layer_features = 304  # 1/4, 1/5, 1/6  Signal = 2_000
    # linear_layer_features = 280  # 1/10  Signal = 1_000
    kernels = [3, 5, 7, 9, 11]
    # kernels = [5, 7, 9, 11, 13]

    seq_length = 50  # Second's
    sampling_rate = 200  # Hz
    nsamples = seq_length * sampling_rate  # Число семплов 10_000
    n_split_samples = 5
    out_samples = nsamples // n_split_samples  # 2_000
    sample_delta = nsamples - out_samples  # 8000
    sample_offset = sample_delta // 2
    multi_validation = True

    train_by_stages = True
    train_by_folds = True

    # 'GPD', 'GRDA', 'LPD', 'LRDA', 'Other', 'Seizure'
    n_stages = 2
    match n_stages:
        case 1:
            train_stages = [0]
            epochs = [100]
            test_total_eval = 2
            total_evals_old = [[(2, 3), (6, 29)]]  # Deprecated
            total_evaluators = [
                [
                    {'band': (2, 2), 'excl_evals': []},
                    {'band': (6, 28), 'excl_evals': []},
                ],
            ]
        case 2:
            train_stages = [0, 1]
            epochs = [50, 100]  # 50,100
            test_total_eval = 4
            total_evals_old = [[(1, 4), (4, 5), (5, 6)], (6, 29)]  # Deprecated
            total_evaluators = [
                [
                    {'band': (1, 3), 'excl_evals': []},
                    {'band': (4, 4), 'excl_evals': ['GPD']},
                    {'band': (5, 5), 'excl_evals': []},
                ],
                [
                    {'band': (6, 28), 'excl_evals': []},
                ],
            ]
        case 3:
            train_stages = [0, 1, 2]
            epochs = [20, 50, 100]
            test_total_eval = 0
            total_evals_old = [(0, 3), (3, 6), (6, 29)]  # Deprecated
            total_evaluators = [
                [
                    {'band': (0, 2), 'excl_evals': []},
                ],
                [
                    {'band': (3, 5), 'excl_evals': []},
                ],
                [
                    {'band': (6, 28), 'excl_evals': []},
                ],
            ]

    n_fold = 10
    train_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # train_folds = [0]

    patience = 11
    seed = 2024

    bandpass_filter = {"low": 0.5, "high": 20, "order": 2}
    rand_filter = {"probab": 0.1, "low": 10, "high": 20, "band": 1.0, "order": 2}
    freq_channels = []  # [(8.0, 12.0)]; [(0.5, 4.5)]
    filter_order = 2

    random_divide_signal = 0.05
    random_close_zone = 0.05
    random_common_negative_signal = 0.0
    random_common_reverse_signal = 0.0
    random_negative_signal = 0.05
    random_reverse_signal = 0.05

    log_step = 100  # Шаг отображения тренировки
    log_show = False

    scheduler = "CosineAnnealingWarmRestarts"  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts','OneCycleLR']

    # CosineAnnealingLR params
    cosanneal_params = {
        "T_max": 6,
        "eta_min": 1e-5,
        "last_epoch": -1,
    }

    # ReduceLROnPlateau params
    reduce_params = {
        "mode": "min",
        "factor": 0.2,
        "patience": 4,
        "eps": 1e-6,
        "verbose": True,
    }

    # CosineAnnealingWarmRestarts params
    cosanneal_res_params = {
        "T_0": 20,
        "eta_min": 1e-6,
        "T_mult": 1,
        "last_epoch": -1,
    }

    target_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    pred_cols = [x + "_pred" for x in target_cols]

    map_features = [
        ("Fp1", "F3"),
        ("Fp2", "F4"),
        ("F3", "C3"),
        ("F4", "C4"),
        ("C3", "P3"),
        ("C4", "P4"),
        ("P3", "O1"),
        ("P4", "O2"),
        ("Fp1", "F7"),
        ("Fp2", "F8"),
        ("F7", "F3"),
        ("F8", "T4"),
        ("T3", "T5"),
        ("T4", "T6"),
        ("T5", "O1"),
        ("T6", "O2"),
        ('Fz', 'Cz'),
        ('Cz', 'Pz'),
    ]

    eeg_features = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8",
                    "T4", "T6", "O2", "EKG"]  # 'Fz', 'Cz', 'Pz'

    # 'F3', 'P3', 'F7', 'T5', 'Fz', 'Cz', 'Pz', 'F4', 'P4', 'F8', 'T6', 'EKG']
    feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}
    simple_features = []  # 'Fz', 'Cz', 'Pz', 'EKG'

    # eeg_features = [row for row in feature_to_index]
    # eeg_feat_size = len(eeg_features)

    n_map_features = len(map_features)
    in_channels = n_map_features + n_map_features * len(freq_channels) + len(simple_features)
    target_size = len(target_cols)

    path_inp = Path("../hms-harmful-brain-activity-classification/input_7")
    path_src = path_inp / "train-raw-csv/"
    file_train = path_inp / "train-raw-csv/train.csv"  # train_small.csv
    path_train = path_src / "train_eegs_small_7"
    file_features_test = path_train / "100261680.parquet"

    file_eeg_specs = "../hms-harmful-brain-activity-classification/output_7/eeg_specs.npy"  # zfg 这个表示的就是原始的eeg预处理过后的

    # file_raw_eeg = path_inp / "brain-eegs/eegs.npy"  # zfg 这个为何没用到？
    # # file_raw_eeg = path_inp / "brain-eegs-plus/eegs.npy"
    # # file_raw_eeg = path_inp / "brain-eegs-full/eegs.npy"

    # if APP.kaggle:
    #     num_workers = 2
    #     parallel = True
    #     # GPU_DEVICES = "auto"


# print(CFG.eeg_feat_size, CFG.in_channels)
print(CFG.feature_to_index)
print(CFG.eeg_features)


def init_logger(log_file=OUTPUT_DIR + "train.log"):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    return mu_x  # quantized


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(
        data, cutoff_freq=20, sampling_rate=CFG.sampling_rate, order=4
):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def denoise_filter(x):
    # Частота дискретизации и желаемые частоты среза (в Гц).
    # Отфильтруйте шумный сигнал
    y = butter_bandpass_filter(x, CFG.lowcut, CFG.highcut, CFG.sampling_rate, order=6)
    y = (y + np.roll(y, -1) + np.roll(y, -2) + np.roll(y, -3)) / 4
    y = y[0:-1:4]
    return y


def eeg_from_parquet(
        parquet_path: str, display: bool = False, seq_length=CFG.seq_length
) -> np.ndarray:
    """
    Эта функция читает файл паркета и извлекает средние 50 секунд показаний. Затем он заполняет значения NaN
    со средним значением (игнорируя NaN).
        :param parquet_path: путь к файлу паркета.
        :param display: отображать графики ЭЭГ или нет.
        :return data: np.array формы (time_steps, eeg_features) -> (10_000, 8)
    """

    # Вырезаем среднюю 50 секундную часть
    eeg = pd.read_parquet(parquet_path, columns=CFG.eeg_features)
    rows = len(eeg)

    # начало смещения данных, чтобы забрать середину
    offset = (rows - CFG.nsamples) // 2

    # средние 50 секунд, имеет одинаковое количество показаний слева и справа
    eeg = eeg.iloc[offset: offset + CFG.nsamples]

    if display:
        plt.figure(figsize=(10, 5))
        offset = 0

    # Конвертировать в numpy

    # создать заполнитель той же формы с нулями
    data = np.zeros((CFG.nsamples, len(CFG.eeg_features)))

    for index, feature in enumerate(CFG.eeg_features):
        x = eeg[feature].values.astype("float32")  # конвертировать в float32

        # Вычисляет среднее арифметическое вдоль указанной оси, игнорируя NaN.
        mean = np.nanmean(x)
        nan_percentage = np.isnan(x).mean()  # percentage of NaN values in feature

        # Заполнение значения Nan
        # Поэлементная проверка на NaN и возврат результата в виде логического массива.
        if nan_percentage < 1:  # если некоторые значения равны Nan, но не все
            x = np.nan_to_num(x, nan=mean)
        else:  # если все значения — Nan
            x[:] = 0
        data[:, index] = x

        if display:
            if index != 0:
                offset += x.max()
            plt.plot(range(CFG.nsamples), x - offset, label=feature)
            offset -= x.min()

    if display:
        plt.legend()
        name = parquet_path.split("/")[-1].split(".")[0]
        plt.yticks([])
        plt.title(f"EEG {name}", size=16)
        plt.show()

    return data


class EEGDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            batch_size: int,
            eegs: Dict[int, np.ndarray],
            mode: str = "train",
            downsample: int = None,
            bandpass_filter: Dict[str, Union[int, float]] = None,
            rand_filter: Dict[str, Union[int, float]] = None,
    ):
        self.df = df
        self.batch_size = batch_size
        self.mode = mode
        self.eegs = eegs
        self.downsample = downsample
        self.offset = None
        self.bandpass_filter = bandpass_filter
        self.rand_filter = rand_filter

    def __len__(self):
        # Обозначает количество пакетов за эпоху
        return len(self.df)

    def __getitem__(self, index):
        # Сгенерировать один пакет данных
        X, y_prob = self.__data_generation(index)
        if self.downsample is not None:
            X = X[:: self.downsample, :]
        output = {
            "eeg": torch.tensor(X, dtype=torch.float32),
            "labels": torch.tensor(y_prob, dtype=torch.float32),
        }
        return output

    def set_offset(self, offset: int):
        self.offset = offset

    def __data_generation(self, index):
        # Генерирует данные, содержащие образцы размера партии
        X = np.zeros(
            (CFG.out_samples, CFG.in_channels), dtype="float32"
        )  # Size=(10000, 14)
        random_divide_signal = False
        row = self.df.iloc[index]  # Строка Pandas
        data = self.eegs[row.eeg_id]  # Size=(10000, 8)
        # zfg data = data[0]
        if CFG.nsamples != CFG.out_samples:
            if self.mode == "train":
                offset = (CFG.sample_delta * random.randint(0, 1000)) // 1000
            elif not self.offset is None:
                offset = self.offset
            else:
                offset = CFG.sample_offset
            if self.mode == "train" and CFG.random_divide_signal > 0.0 and random.uniform(0.0,
                                                                                          1.0) <= CFG.random_divide_signal:
                random_divide_signal = True
                multipliers = [(1, 2), (2, 3), (3, 4), (3, 5)]
                koef_1, koef_2 = multipliers[random.randint(0, 3)]
                offset = (koef_1 * offset) // koef_2
                data = data[offset:offset + (CFG.out_samples * koef_2) // koef_1, :]
            else:
                data = data[offset:offset + CFG.out_samples, :]  # data = data
        reverse_signal = False
        negative_signal = False
        if self.mode == "train":
            if CFG.random_common_reverse_signal > 0.0 and random.uniform(0.0, 1.0) <= CFG.random_common_reverse_signal:
                reverse_signal = True
            if CFG.random_common_negative_signal > 0.0 and random.uniform(0.0,
                                                                          1.0) <= CFG.random_common_negative_signal:
                negative_signal = True
        for i, (feat_a, feat_b) in enumerate(CFG.map_features):
            if self.mode == "train" and CFG.random_close_zone > 0.0 and random.uniform(0.0,
                                                                                       1.0) <= CFG.random_close_zone:
                continue
            diff_feat = (
                    data[:, CFG.feature_to_index[feat_a]]
                    - data[:, CFG.feature_to_index[feat_b]]
            )  # Size=(10000,)

            if self.mode == "train":
                if reverse_signal or CFG.random_reverse_signal > 0.0 and random.uniform(0.0,
                                                                                        1.0) <= CFG.random_reverse_signal:
                    diff_feat = np.flip(diff_feat)
                if negative_signal or CFG.random_negative_signal > 0.0 and random.uniform(0.0,
                                                                                          1.0) <= CFG.random_negative_signal:
                    diff_feat = -diff_feat
            if not self.bandpass_filter is None:
                diff_feat = butter_bandpass_filter(
                    diff_feat,
                    self.bandpass_filter["low"],
                    self.bandpass_filter["high"],
                    CFG.sampling_rate,
                    order=self.bandpass_filter["order"],
                )
            if random_divide_signal:
                diff_feat = scisig.upfirdn([1.0, 1, 1.0], diff_feat, koef_1, koef_2)  # linear interp, rate 2/3
                diff_feat = diff_feat[0:CFG.out_samples]
            if (
                    self.mode == "train"
                    and not self.rand_filter is None
                    and random.uniform(0.0, 1.0) <= self.rand_filter["probab"]
            ):
                lowcut = random.randint(
                    self.rand_filter["low"], self.rand_filter["high"]
                )
                highcut = lowcut + self.rand_filter["band"]
                diff_feat = butter_bandpass_filter(
                    diff_feat,
                    lowcut,
                    highcut,
                    CFG.sampling_rate,
                    order=self.rand_filter["order"],
                )
            X[:, i] = diff_feat
        n = CFG.n_map_features
        if len(CFG.freq_channels) > 0:
            for i in range(CFG.n_map_features):
                diff_feat = X[:, i]
                for j, (lowcut, highcut) in enumerate(CFG.freq_channels):
                    band_feat = butter_bandpass_filter(
                        diff_feat, lowcut, highcut, CFG.sampling_rate, order=CFG.filter_order,  # 6
                    )
                    X[:, n] = band_feat
                    n += 1
        for spml_feat in CFG.simple_features:
            feat_val = data[:, CFG.feature_to_index[spml_feat]]
            if not self.bandpass_filter is None:
                feat_val = butter_bandpass_filter(
                    feat_val,
                    self.bandpass_filter["low"],
                    self.bandpass_filter["high"],
                    CFG.sampling_rate,
                    order=self.bandpass_filter["order"],
                )
            if (
                    self.mode == "train"
                    and not self.rand_filter is None
                    and random.uniform(0.0, 1.0) <= self.rand_filter["probab"]
            ):
                lowcut = random.randint(
                    self.rand_filter["low"], self.rand_filter["high"]
                )
                highcut = lowcut + self.rand_filter["band"]
                feat_val = butter_bandpass_filter(
                    feat_val,
                    lowcut,
                    highcut,
                    CFG.sampling_rate,
                    order=self.rand_filter["order"],
                )
            X[:, n] = feat_val
            n += 1
        X = np.clip(X, -1024, 1024)
        X = np.nan_to_num(X, nan=0) / 32.0
        X = butter_lowpass_filter(X, order=CFG.filter_order)  # 4
        y_prob = np.zeros(CFG.target_size, dtype="float32")  # Size=(6,)
        if self.mode != "test":
            y_prob = row[CFG.target_cols].values.astype(np.float32)
        return X, y_prob


class KLDivLossWithLogits(nn.KLDivLoss):
    def __init__(self):
        super().__init__(reduction="batchmean")

    def forward(self, y, t):
        y = nn.functional.log_softmax(y, dim=1)
        loss = super().forward(y, t)
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True  # Это опция требует много паямяти GPU
    # pl.seed_everything(seed)


class ResNet_1D_Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            downsampling,
            dilation=1,
            groups=1,
            dropout=0.0,
    ):
        super(ResNet_1D_Block, self).__init__()

        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        # self.relu = nn.ReLU(inplace=False)
        # self.relu_1 = nn.PReLU()
        # self.relu_2 = nn.PReLU()
        self.relu_1 = nn.Hardswish()
        self.relu_2 = nn.Hardswish()

        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.maxpool = nn.MaxPool1d(
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=dilation,
        )
        self.downsampling = downsampling

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu_1(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.maxpool(out)
        identity = self.downsampling(x)

        out += identity
        return out


class EEGNet(nn.Module):
    def __init__(
            self,
            kernels,
            in_channels,
            fixed_kernel_size,
            num_classes,
            linear_layer_features,
            dilation=1,
            groups=1,
    ):
        super(EEGNet, self).__init__()
        self.kernels = kernels
        self.planes = 24
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels

        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.planes,
                kernel_size=(kernel_size),
                stride=1,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        # self.relu = nn.ReLU(inplace=False)
        # self.relu_1 = nn.ReLU()
        # self.relu_2 = nn.ReLU()
        self.relu_1 = nn.SiLU()
        self.relu_2 = nn.SiLU()

        self.conv1 = nn.Conv1d(
            in_channels=self.planes,
            out_channels=self.planes,
            kernel_size=fixed_kernel_size,
            stride=2,
            padding=2,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.block = self._make_resnet_layer(
            kernel_size=fixed_kernel_size,
            stride=1,
            dilation=dilation,
            groups=groups,
            padding=fixed_kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)

        self.rnn = nn.GRU(
            input_size=self.in_channels,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            # dropout=0.2,
        )

        self.fc = nn.Linear(in_features=linear_layer_features, out_features=num_classes)

    def _make_resnet_layer(
            self,
            kernel_size,
            stride,
            dilation=1,
            groups=1,
            blocks=9,
            padding=0,
            dropout=0.0,
    ):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            downsampling = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            )
            layers.append(
                ResNet_1D_Block(
                    in_channels=self.planes,
                    out_channels=self.planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    downsampling=downsampling,
                    dilation=dilation,
                    groups=groups,
                    dropout=dropout,
                )
            )
        return nn.Sequential(*layers)

    def extract_features(self, x):
        x = x.permute(0, 2, 1)

        out_sep = []
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu_1(out)
        out = self.conv1(out)

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.avgpool(out)

        out = out.reshape(out.shape[0], -1)
        rnn_out, _ = self.rnn(x.permute(0, 2, 1))
        new_rnn_h = rnn_out[:, -1, :]

        new_out = torch.cat([out, new_rnn_h], dim=1)
        return new_out

    def forward(self, x):
        new_out = self.extract_features(x)
        result = self.fc(new_out)
        return result


class Adan(Optimizer):
    """
    Implements a pytorch variant of Adan
    Adan was proposed in
    Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models[J]. arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float, flot], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.98, 0.92, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay (L2 penalty) (default: 0)
        max_grad_norm (float, optional): value used to clip
            global grad norm (default: 0.0 no clip)
        no_prox (bool): how to perform the decoupled weight decay (default: False)
    """

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.98, 0.92, 0.99),
            eps=1e-8,
            weight_decay=0.2,
            max_grad_norm=0.0,
            no_prox=False,
    ):
        if not 0.0 <= max_grad_norm:
            raise ValueError("Invalid Max grad norm: {}".format(max_grad_norm))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            no_prox=no_prox,
        )
        super(Adan, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adan, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("no_prox", False)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group["step"] = 0
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # Exponential moving average of gradient difference
                    state["exp_avg_diff"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step.
        """
        if self.defaults["max_grad_norm"] > 0:
            device = self.param_groups[0]["params"][0].device
            global_grad_norm = torch.zeros(1, device=device)

            max_grad_norm = torch.tensor(self.defaults["max_grad_norm"], device=device)
            for group in self.param_groups:

                for p in group["params"]:
                    if p.grad is not None:
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm)

            clip_global_grad_norm = torch.clamp(
                max_grad_norm / (global_grad_norm + group["eps"]), max=1.0
            )
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            bias_correction1 = 1.0 - beta1 ** group["step"]
            bias_correction2 = 1.0 - beta2 ** group["step"]
            bias_correction3 = 1.0 - beta3 ** group["step"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["exp_avg_diff"] = torch.zeros_like(p)

                grad = p.grad.mul_(clip_global_grad_norm)
                if "pre_grad" not in state or group["step"] == 1:
                    state["pre_grad"] = grad

                copy_grad = grad.clone()

                exp_avg, exp_avg_sq, exp_avg_diff = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["exp_avg_diff"],
                )
                diff = grad - state["pre_grad"]

                update = grad + beta2 * diff
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
                exp_avg_diff.mul_(beta2).add_(diff, alpha=1 - beta2)  # diff_t
                exp_avg_sq.mul_(beta3).addcmul_(update, update, value=1 - beta3)  # n_t

                denom = ((exp_avg_sq).sqrt() / math.sqrt(bias_correction3)).add_(
                    group["eps"]
                )
                update = (
                    (
                            exp_avg / bias_correction1
                            + beta2 * exp_avg_diff / bias_correction2
                    )
                ).div_(denom)

                if group["no_prox"]:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                else:
                    p.add_(update, alpha=-group["lr"])
                    p.data.div_(1 + group["lr"] * group["weight_decay"])

                state["pre_grad"] = copy_grad


def train_fn(
        stage, fold, train_loader, model, criterion, optimizer, epoch, scheduler, device
):
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0

    for step, batch in enumerate(train_loader):
        eegs = batch["eeg"].to(device)
        labels = batch["labels"].to(device)
        batch_size = labels.size(0)

        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(eegs)
            loss = criterion(F.log_softmax(y_preds, dim=1), labels)

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        losses.update(loss.item(), batch_size)

        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm
        )

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()

        if CFG.log_show and (
                step % CFG.log_step == 0 or step == (len(train_loader) - 1)
        ):
            # remain=timeSince(start, float(step + 1) / len(train_loader))
            LOGGER.info(
                f"Epoch {epoch + 1} [{step}/{len(train_loader)}] Loss: {losses.val:.4f} Loss Avg:{losses.avg:.4f}"
            )
            # "Elapsed {remain:s} Grad: {grad_norm:.4f}  LR: {cheduler.get_lr()[0]:.8f}"

        if CFG.wandb:
            wandb.log(
                {
                    f"[fold{fold}] loss": losses.val,
                    f"[fold{fold}] lr": scheduler.get_lr()[0],
                }
            )
    return losses.avg


def valid_fn(stage, epoch, valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    targets = []
    start = end = time.time()

    for step, batch in enumerate(valid_loader):
        eegs = batch["eeg"].to(device)
        labels = batch["labels"].to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(eegs)
            loss = criterion(F.log_softmax(y_preds, dim=1), labels)

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        losses.update(loss.item(), batch_size)
        preds.append(nn.Softmax(dim=1)(y_preds).to("cpu").numpy())
        targets.append(labels.to("cpu").numpy())
        end = time.time()

        if CFG.log_show and (
                step % CFG.log_step == 0 or step == (len(valid_loader) - 1)
        ):
            # remain=timeSince(start, float(step + 1) / len(valid_loader))
            LOGGER.info(
                f"Epoch {epoch + 1} VALIDATION: [{step}/{len(valid_loader)}] Val Loss: {losses.val:.4f} Val Loss Avg: {losses.avg:.4f}"
            )
            # Elapsed {remain:s}

    predictions = np.concatenate(preds)
    targets = np.concatenate(targets)

    return losses.avg, predictions


def build_optimizer(cfg, model, device, epochs, num_batches_per_epoch):
    lr = cfg.lr
    # lr = default_configs["lr"]
    if cfg.optimizer == "SAM":
        base_optimizer = (
            torch.optim.SGD
        )  # define an optimizer for the "sharpness-aware" update
        optimizer_model = SAM(
            model.parameters(),
            base_optimizer,
            lr=lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
            adaptive=True,
        )
    elif cfg.optimizer == "Ranger21":
        optimizer_model = Ranger21(
            model.parameters(),
            lr=lr,
            weight_decay=cfg.weight_decay,
            num_epochs=epochs,
            num_batches_per_epoch=num_batches_per_epoch,
        )
    elif cfg.optimizer == "SGD":
        optimizer_model = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=cfg.weight_decay, momentum=0.9
        )
    elif cfg.optimizer == "Adam":
        optimizer_model = Adam(model.parameters(), lr=lr, weight_decay=CFG.weight_decay)
    elif cfg.optimizer == "AdamW":
        optimizer_model = AdamW(
            model.parameters(), lr=lr, weight_decay=CFG.weight_decay
        )
    elif cfg.optimizer == "Lion":
        optimizer_model = Lion(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "Adan":
        optimizer_model = Adan(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    return optimizer_model


def get_scheduler(optimizer, epochs, steps_per_epoch):
    if CFG.scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, **CFG.reduce_params)
    elif CFG.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, **CFG.cosanneal_params)
    elif CFG.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, **CFG.cosanneal_res_params)
    elif CFG.scheduler == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer=optimizer,
            epochs=epochs,
            pct_start=0.0,
            steps_per_epoch=steps_per_epoch,
            max_lr=CFG.lr,
            div_factor=25,
            final_div_factor=4.0e-01,
        )
    return scheduler


def train_loop(stage, epochs, folds, fold, directory, prev_dir, eggs):
    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values

    train_dataset = EEGDataset(
        train_folds,
        batch_size=CFG.batch_size,
        mode="train",
        eegs=eggs,
        bandpass_filter=CFG.bandpass_filter,
        rand_filter=CFG.rand_filter,
    )

    valid_dataset = EEGDataset(
        valid_folds,
        batch_size=CFG.batch_size,
        mode="valid",
        eegs=eggs,
        bandpass_filter=CFG.bandpass_filter,
        # rand_filter=CFG.rand_filter,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size * CFG.batch_koef_valid,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    LOGGER.info(
        f"========== stage: {stage} fold: {fold} training {len(train_loader)} / {len(valid_loader)} =========="
    )

    model = EEGNet(
        kernels=CFG.kernels,
        in_channels=CFG.in_channels,
        fixed_kernel_size=CFG.fixed_kernel_size,
        num_classes=CFG.target_size,
        linear_layer_features=CFG.linear_layer_features,
    )

    if stage > 1:
        model_weight = f"{prev_dir}{CFG.model_name}_ver-{CFG.VERSION}_stage-{stage - 1}_fold-{fold}_best.pth"
        checkpoint = torch.load(model_weight, map_location=device)
        model.load_state_dict(checkpoint["model"])

    model.to(device)

    # CPMP: wrap the model to use all GPUs
    if CFG.parallel:
        model = nn.DataParallel(model)

    optimizer = build_optimizer(
        CFG, model, device, epochs=epochs, num_batches_per_epoch=len(train_loader)
    )
    scheduler = get_scheduler(
        optimizer, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    criterion = nn.KLDivLoss(reduction="batchmean")

    best_score = np.inf
    for epoch in range(epochs):
        start_time = time.time()

        # train
        avg_loss = train_fn(
            stage,
            fold,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            device,
        )

        # eval
        valid_dataset.set_offset(CFG.sample_offset)
        avg_val_loss, predictions = valid_fn(
            stage,
            epoch,
            valid_loader,
            model,
            criterion,
            device,
        )

        avg_loss_line = ''
        if CFG.multi_validation:
            multi_avg_val_loss = np.zeros(CFG.n_split_samples)
            start = (2 * CFG.sample_delta) // CFG.n_split_samples
            finish = (3 * CFG.sample_delta) // CFG.n_split_samples
            delta = (finish - start) // 5
            for i in range(CFG.n_split_samples):
                valid_dataset.set_offset(start)
                multi_avg_val_loss[i], _ = valid_fn(
                    stage,
                    epoch,
                    valid_loader,
                    model,
                    criterion,
                    device,
                )
                avg_loss_line += f" {multi_avg_val_loss[i]:.4f}"
                start += delta
            avg_loss_line += f" mean={np.mean(multi_avg_val_loss):.4f}"

        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch {epoch + 1} Avg Train Loss: {avg_loss:.4f} Avg Valid Loss: {avg_val_loss:.4f} / {avg_loss_line}"
        )
        #   time: {elapsed:.0f}s
        if CFG.wandb:
            wandb.log(
                {
                    f"[fold{fold}] stage": stage,
                    f"[fold{fold}] epoch": epoch + 1,
                    f"[fold{fold}] avg_train_loss": avg_loss,
                    f"[fold{fold}] avg_val_loss": avg_val_loss,
                    # f"[fold{fold}] score": score,
                }
            )

        if CFG.save_all_models:
            torch.save(
                {"model": model.module.state_dict(), "predictions": predictions},
                f"{directory}{CFG.model_name}_ver-{CFG.VERSION}_stage-{stage}_fold-{fold}_epoch-{epoch}_val-{avg_val_loss:.4f}_train-{avg_loss:.4f}.pth",
            )

        if best_score > avg_val_loss:
            best_score = avg_val_loss
            LOGGER.info(f"Epoch {epoch + 1} Save Best Valid Loss: {avg_val_loss:.4f}")
            # CPMP: save the original model. It is stored as the module attribute of the DP model.
            torch.save(
                {"model": model.module.state_dict(), "predictions": predictions},
                f"{directory}{CFG.model_name}_ver-{CFG.VERSION}_stage-{stage}_fold-{fold}_best.pth",
            )

    predictions = torch.load(
        f"{directory}{CFG.model_name}_ver-{CFG.VERSION}_stage-{stage}_fold-{fold}_best.pth",
        map_location=torch.device("cpu"),
    )["predictions"]

    # valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions
    valid_folds[CFG.pred_cols] = predictions
    valid_folds[CFG.target_cols] = valid_labels

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds, best_score


train = pd.read_csv(CFG.file_train)

TARGETS = train.columns[-6:]
print("Train shape:", train.shape)
print("Targets", list(TARGETS))

train["total_evaluators"] = train[CFG.target_cols].sum(axis=1)

train_uniq = train.drop_duplicates(subset=["eeg_id"] + list(TARGETS))

print(f"There are {train.patient_id.nunique()} patients in the training data.")
print(f"There are {train.eeg_id.nunique()} EEG IDs in the training data.")
print(f"There are {train_uniq.shape[0]} unique eeg_id + votes in the training data.")

if CFG.visualize:
    train_uniq.eeg_id.value_counts().value_counts().plot(
        kind="bar",
        title=f"Distribution of Count of EEG w Unique Vote: "
              f"{train_uniq.shape[0]} examples",
    )

del train_uniq
_ = gc.collect()

if CFG.visualize:
    plt.figure(figsize=(10, 6))
    plt.hist(train["total_evaluators"], bins=10, color="blue", edgecolor="black")
    plt.title("Histogram of Total Evaluators")
    plt.xlabel("Total Evaluators")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# 打印数
print(train.shape[0])

# zfg 在train 中挑选出数据
# 定义文件路径和目标列名
file_train = "../hms-harmful-brain-activity-classification/input_7/train-raw-csv/train_small.csv"
target_column = "label_id"

# 读取train_small.csv文件并获取eeg_id列的所有数据
label_ids = pd.read_csv(file_train)[target_column].tolist()
print(f"Read {len(label_ids)} label_ids from {file_train}")

# 打印行数
print("Number of rows in train:", train.shape[0])

# 检查eeg_id列是否存在
if target_column in train.columns:
    print(f"{target_column} column exists")
else:
    raise ValueError(f"{target_column} column does not exist in the DataFrame. Columns are: {train.columns.tolist()}")

# 使用eeg_ids列表进行索引匹配
train_sample = train[train[target_column].isin(label_ids)]
print(f"Number of rows in train_sample: {train_sample.shape[0]}")
train = train_sample
# zfg


y_data = train[TARGETS].values + 0.166666667  # Regularization value
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

train["target"] = train["expert_consensus"]

train[train['total_evaluators'] == CFG.test_total_eval].groupby(['expert_consensus', 'total_evaluators']).count()
if CFG.test_total_eval > 0:
    train['key_id'] = range(train.shape[0])

    train_pop_olds = []
    for total_eval in CFG.total_evals_old:
        if type(total_eval) is list:
            pop_idx = (train["total_evaluators"] >= total_eval[0][0]) & (
                    train["total_evaluators"] < total_eval[0][1]
            ) | (train["total_evaluators"] >= total_eval[1][0]) & (
                              train["total_evaluators"] < total_eval[1][1]
                      )
        else:
            pop_idx = (train["total_evaluators"] >= total_eval[0]) & (
                    train["total_evaluators"] < total_eval[1]
            )

        train_pop = train[pop_idx].copy().reset_index()

        sgkf = GroupKFold(n_splits=CFG.n_fold)
        train_pop["fold"] = -1
        for fold_id, (_, val_idx) in enumerate(
                sgkf.split(train_pop, y=train_pop["target"], groups=train_pop["patient_id"])
        ):
            train_pop.loc[val_idx, "fold"] = fold_id

        train_pop_olds.append(train_pop)
        print(train_pop.shape[0])

train_pops = []
for eval_list in CFG.total_evaluators:
    result = []
    train_pop = train
    for eval_dict in eval_list:
        band = eval_dict['band']
        pop_idx = (train_pop["total_evaluators"] >= band[0])
        pop_idx &= (train_pop["total_evaluators"] <= band[1])
        for exclude in eval_dict['excl_evals']:
            pop_idx &= ~(train_pop['expert_consensus'] == exclude)
            pass
        result.append(train_pop[pop_idx])
    train_pop = pd.concat(result).copy().reset_index()

    sgkf = GroupKFold(n_splits=CFG.n_fold)
    train_pop["fold"] = -1
    for fold_id, (_, val_idx) in enumerate(
            sgkf.split(train_pop, y=train_pop["target"], groups=train_pop["patient_id"])
    ):
        train_pop.loc[val_idx, "fold"] = fold_id

    train_pops.append(train_pop)
    print(train_pop.shape[0])

train_0 = train_pops[0]
train_0[train_0['total_evaluators'] == CFG.test_total_eval].groupby(['expert_consensus', 'total_evaluators']).count()

if CFG.test_total_eval > 0:
    df_old = train_pop_olds[0].copy(deep=True).set_index(['key_id'], drop=True).drop(columns=['fold'])
    df_new = train_pops[0].copy(deep=True).set_index(['key_id'], drop=True).drop(columns=['fold'])

    # outer merge the two DataFrames, adding an indicator column called 'Exist'
    diff_df = pd.merge(df_old, df_new, how='outer', indicator='Exist')

    # find which rows don't exist in both DataFrames
    diff_df = diff_df.loc[diff_df['Exist'] != 'both']
    # display(diff_df)

    del df_old, df_new, diff_df, train_pop_olds
    _ = gc.collect()

if CFG.visualize:
    print("Pop 1: train unique eeg_id + votes shape:", train_pops[0].shape)
    plt.figure(figsize=(10, 6))
    plt.hist(train["total_evaluators"], bins=10, color="blue", edgecolor="black")
    plt.title("Histogram of Total Evaluators")
    plt.xlabel("Total Evaluators")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# del all_eeg_specs
_ = gc.collect()

if CFG.create_eegs:
    all_eegs = {}
    visualize = 1 if CFG.visualize else 0
    eeg_ids = train.eeg_id.unique()

    for i, eeg_id in tqdm(enumerate(eeg_ids)):

        # Сохранить ЭЭГ в словаре Python для массивов numpy
        eeg_path = CFG.path_train / f"{eeg_id}.parquet"

        # Вырезаем среднюю 50 секундную часть и заполняем по среднему Nan
        data = eeg_from_parquet(eeg_path, display=i < visualize)
        all_eegs[eeg_id] = data  # zfg 键值对

        if i == visualize:
            if CFG.create_eegs:
                print(
                    f"Processing {train['eeg_id'].nunique()} eeg parquets... ", end=""
                )
            else:
                print(f"Reading {len(eeg_ids)} eeg NumPys from disk.")
                break
    np.save("../hms-harmful-brain-activity-classification/output_7/eeg_specs.npy", all_eegs)
    # 加载并打印 numpy 文件的键
    loaded_eegs = np.load("../hms-harmful-brain-activity-classification/output_7/eeg_specs.npy",
                          allow_pickle=True).item()
    print("Keys in eeg_specs.npy:", loaded_eegs.keys())
    # file_eeg_specs =  "/media/laplace/CA4960E6F5A4B1BD/hms-harmful-brain-activity-classification/output_7/eeg_specs.npy"

else:
    all_eegs = np.load('../input/raw-data-20183-50/eeg_raw_20183_50s.npy', allow_pickle=True).item()

if CFG.visualize:
    frequencies = [1, 2, 4, 8, 16][::-1]  # frequencies in Hz
    x = [all_eegs[eeg_ids[0]][:, 0]]  # select one EEG feature

    for frequency in frequencies:
        x.append(butter_lowpass_filter(x[0], cutoff_freq=frequency))

    plt.figure(figsize=(12, 8))
    plt.plot(range(CFG.nsamples), x[0], label="without filter")
    for k in range(1, len(x)):
        plt.plot(
            range(CFG.nsamples),
            x[k] - k * (x[0].max() - x[0].min()),
            label=f"with filter {frequencies[k - 1]}Hz",
        )

    plt.legend()
    plt.yticks([])
    plt.title("Butter Low-Pass Filter Examples", size=18)
    plt.show()

import numpy as np

max_int_value = (1 << 31) - 1

keys_to_remove = []

for key, value_list in list(all_eegs.items()):
    if key <= 0:
        truncated_value = key
        n = 32  # 假设为32位整数
        # 原始值 = 截断值 + 2^(n)
        original_value = truncated_value + (1 << n)
        keys_to_remove.append(key)
        all_eegs[original_value] = value_list

# 在迭代结束后删除需要删除的键
for key in keys_to_remove:
    del all_eegs[key]

# all_eegs[1628180742].shape

if CFG.visualize:
    train_dataset = EEGDataset(
        train_pops[0], batch_size=CFG.batch_size, eegs=all_eegs, mode="train"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    output = train_dataset[0]
    X, y = output["eeg"], output["labels"]
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    iot = torch.randn(2, CFG.nsamples, CFG.in_channels)  # .cuda()
    model = EEGNet(
        kernels=CFG.kernels,
        in_channels=CFG.in_channels,
        fixed_kernel_size=CFG.fixed_kernel_size,
        num_classes=CFG.target_size,
        linear_layer_features=CFG.linear_layer_features,
    )
    output = model(iot)
    print(output.shape)

    for batch in train_loader:
        X = batch.pop("eeg")
        y = batch.pop("labels")
        for item in range(4):
            plt.figure(figsize=(20, 4))
            offset = 0
            for col in range(X.shape[-1]):
                if col != 0:
                    offset -= X[item, :, col].min()
                plt.plot(
                    range(CFG.nsamples),
                    X[item, :, col] + offset,
                    label=f"feature {col + 1}",
                )
                offset += X[item, :, col].max()
            tt = f"{y[col][0]:0.1f}"
            for t in y[col][1:]:
                tt += f", {t:0.1f}"
            plt.title(f"EEG_Id = {eeg_ids[item]}\nTarget = {tt}", size=14)
            plt.legend()
            plt.show()
        break

    del iot, model
    gc.collect()


def get_score(preds, targets):
    oof = pd.DataFrame(preds.copy())
    oof["id"] = np.arange(len(oof))
    true = pd.DataFrame(targets.copy())
    true["id"] = np.arange(len(true))
    cv = kaggle_kl_div.score(solution=true, submission=oof, row_id_column_name="id")
    return cv


def get_result(result_df):
    gt = result_df[["eeg_id"] + CFG.target_cols]
    gt.sort_values(by="eeg_id", inplace=True)
    gt.reset_index(inplace=True, drop=True)
    preds = result_df[["eeg_id"] + CFG.pred_cols]
    preds.columns = ["eeg_id"] + CFG.target_cols
    preds.sort_values(by="eeg_id", inplace=True)
    preds.reset_index(inplace=True, drop=True)
    score_loss = get_score(gt[CFG.target_cols], preds[CFG.target_cols])
    LOGGER.info(f"Score with best loss weights: {score_loss}")


if __name__ == "__main__" and CFG.train_by_stages:
    seed_torch(seed=CFG.seed)

    prev_dir = ""
    for stage in range(len(CFG.total_evaluators)):
        pop_dir = f"{OUTPUT_DIR}pop_{stage + 1}_weight_oof/"
        if not os.path.exists(pop_dir):
            os.makedirs(pop_dir)

        if stage not in CFG.train_stages:
            prev_dir = pop_dir
            continue

        oof_df = pd.DataFrame()
        scores = []
        for fold in CFG.train_folds:
            train_oof_df, score = train_loop(
                stage=stage + 1,
                epochs=CFG.epochs[stage],
                fold=fold,
                folds=train_pops[stage],
                directory=pop_dir,
                prev_dir=prev_dir,
                eggs=all_eegs,
            )

            oof_df = pd.concat([oof_df, train_oof_df])
            scores.append(score)

            LOGGER.info(f"========== stage: {stage + 1} fold: {fold} result ==========")
            LOGGER.info(f"Score with best loss weights stage{stage + 1}: {score:.4f}")

        LOGGER.info(f"==================== CV ====================")
        LOGGER.info(f"Score with best loss weights: {np.mean(scores):.4f}")

        oof_df.reset_index(drop=True, inplace=True)
        oof_df.to_csv(
            f"{pop_dir}{CFG.model_name}_oof_df_ver-{CFG.VERSION}_stage-{stage + 1}.csv",
            index=False,
        )

        prev_dir = pop_dir

    if CFG.wandb:
        wandb.finish()

if __name__ == "__main__" and CFG.train_by_folds:
    seed_torch(seed=CFG.seed)

    stages_scores = {i: [] for i in CFG.train_stages}
    stages_oof_df = {i: pd.DataFrame() for i in CFG.train_stages}

    for fold in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:

        prev_dir = ""
        for stage in range(len(CFG.total_evaluators)):

            pop_dir = f"{OUTPUT_DIR}pop_{stage + 1}_weight_oof/"
            if not os.path.exists(pop_dir):
                os.makedirs(pop_dir)

            if stage not in CFG.train_stages:
                prev_dir = pop_dir
                continue

            train_oof_df, score = train_loop(
                stage=stage + 1,
                epochs=CFG.epochs[stage],
                fold=fold,
                folds=train_pops[stage],
                directory=pop_dir,
                prev_dir=prev_dir,
                eggs=all_eegs,
            )

            stages_oof_df[stage] = pd.concat([stages_oof_df[stage], train_oof_df])
            stages_scores[stage].append(score)

            prev_dir = pop_dir

            LOGGER.info(f"========== fold: {fold} stage: {stage + 1} result ==========")
            LOGGER.info(f"Score with best loss weights stage{stage + 1}: {score:.4f}")

    for stage, scores in stages_scores.items():
        LOGGER.info(f"============ CV score with best loss weights ============")
        LOGGER.info(f"Stage {stage}: {np.mean(scores):.4f}")

    for stage, oof_df in stages_oof_df.items():
        pop_dir = f"{OUTPUT_DIR}pop_{stage + 1}_weight_oof/"
        oof_df.reset_index(drop=True, inplace=True)
        oof_df.to_csv(
            f"{pop_dir}{CFG.model_name}_oof_df_ver-{CFG.VERSION}_stage-{stage + 1}.csv",
            index=False,
        )

    if CFG.wandb:
        wandb.finish()

# === Pre-process OOF ===
gt = oof_df[["eeg_id"] + CFG.target_cols]
gt.sort_values(by="eeg_id", inplace=True)
gt.reset_index(inplace=True, drop=True)

preds = oof_df[["eeg_id"] + CFG.pred_cols]
preds.columns = ["eeg_id"] + CFG.target_cols
preds.sort_values(by="eeg_id", inplace=True)
preds.reset_index(inplace=True, drop=True)

y_trues = gt[CFG.target_cols]
y_preds = preds[CFG.target_cols]

oof = pd.DataFrame(y_preds.copy())
oof["id"] = np.arange(len(oof))

true = pd.DataFrame(y_trues.copy())
true["id"] = np.arange(len(true))

cv = kaggle_kl_div.score(solution=true, submission=oof, row_id_column_name="id")
print(f"CV Score with resnet1D_gru Raw EEG = {cv:.4f}")
