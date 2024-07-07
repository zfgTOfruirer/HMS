# zfg 模型1-8的推理，融合
import gc
import os
import typing as tp
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.notebook import tqdm

# zfg model1 + model2

pretrained_cfg_overlay = {
    'file': r"/home/laplace/.cache/huggingface/hub/models--timm--tf_efficientnetv2_s.in21k_ft_in1k/pytorch_model.bin"}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT = Path("../hms-harmful-brain-activity-classification")
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
DATA = INPUT / "train-raw-csv"
TEST_SPEC = DATA / "test_spectrograms"
TMP = ROOT / "tmp"
TEST_SPEC_SPLIT = TMP / "train_spectrograms_split"
TMP.mkdir(exist_ok=True)
TEST_SPEC_SPLIT.mkdir(exist_ok=True)

RANDAM_SEED = 1086
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
N_CLASSES = len(CLASSES)
FOLDS = [0, 1, 2, 3, 4]
N_FOLDS = len(FOLDS)
test = pd.read_csv(
    "../hms-harmful-brain-activity-classification/input/train-raw-csv/test.csv")

# zfg 下面为 eeg to spec

m = 0
for spec_id in test["spectrogram_id"]:
    spec = pd.read_parquet(TEST_SPEC / f"{spec_id}.parquet")
    spec_arr = spec.fillna(0).values[:, 1:].T.astype("float32")  # 所有行的第一列到最后一列缺失数值填充

    # 检查形状是否为 (400, 300)，如果不是，进行适当调整
    if spec_arr.shape != (400, 300):
        # 根据实际需要调整代码，这里只是一个示例
        resized_spec_arr = np.zeros((400, 300), dtype='float32')
        # 调整大小，复制数据到新数组中
        h, w = spec_arr.shape

        m += 1
        resized_spec_arr[:min(400, h), :min(300, w)] = spec_arr[:min(400, h), :min(300, w)]
        spec_arr = resized_spec_arr  # 更新 spec_arr 为调整后的数组

    np.save(TEST_SPEC_SPLIT / f"{spec_id}.npy", spec_arr)

print(f"{m}")

im = np.load(
    "../hms-harmful-brain-activity-classification/tmp/train_spectrograms_split/67817419.npy")
print(im.shape)

test = pd.read_csv(
    '../hms-harmful-brain-activity-classification/input/train-raw-csv/test.csv')
print('Test shape', test.shape)


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet='haar', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    ret = pywt.waverec(coeff, wavelet, mode='per')
    return ret


USE_WAVELET = None
NAMES = ['LL', 'LP', 'RP', 'RR']
FEATS = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
         ['Fp1', 'F3', 'C3', 'P3', 'O1'],
         ['Fp2', 'F8', 'T4', 'T6', 'O2'],
         ['Fp2', 'F4', 'C4', 'P4', 'O2']]


def spectrogram_from_eeg(parquet_path, display=False):
    # 从 parquet_path 中加载中间50秒的EEG数据
    eeg = pd.read_parquet(parquet_path)

    # 定义变量以存储频谱图
    img = np.zeros((128, 256, 4), dtype='float32')

    # 如果需要显示频谱图，则显示
    if display: plt.figure(figsize=(10, 7))

    # 对每个通道计算差异
    for k in range(4):
        COLS = FEATS[k]

        # 计算对角线上的差异
        for kk in range(4):

            # 计算成对差异
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            # 填充NaN
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            # 去噪
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)

            # 添加到信号列表
            signals.append(x)

            # 计算原始频谱图
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x) // 256,
                                                      n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            # 对数变换
            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :width]

            # 标准化到-1到1
            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db

        # 对4个montage差异求平均值
        img[:, :, k] /= 4.0

        # 如果需要显示频谱图，则显示
        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect='auto', origin='lower')
            plt.title(f'EEG {eeg_id} - Spectrogram {NAMES[k]}')

    # 如果需要显示频谱图，则显示
    if display:
        plt.show()
        plt.figure(figsize=(10, 5))
        offset = 0
        for k in range(4):
            if k > 0: offset -= signals[3 - k].min()
            plt.plot(range(10_000), signals[k] + offset, label=NAMES[3 - k])
            offset += signals[3 - k].max()
        plt.legend()
        plt.title(f'EEG {eeg_id} Signals')
        plt.show()
        print()
        print('#' * 25)
        print()

    # 返回频谱图
    return img


# 从CSV文件中读取EEG频谱图
PATH2 = '../hms-harmful-brain-activity-classification/input/train-raw-csv/test_eegs/'  # 指定测试数据文件夹路径
DISPLAY = 1  # 设置显示频谱图的数量
EEG_IDS2 = test.eeg_id.unique()  # 获取测试数据中所有EEG的ID
all_eegs = {}  # 定义一个空字典，用于存储所有EEG的频谱图

print('将EEG转换为频谱图...');  # 打印提示信息，开始将EEG转换为频谱图
print()  # 打印换行符

for i, eeg_id in enumerate(EEG_IDS2):  # 遍历所有EEG ID
    # 从给定的EEG文件创建频谱图，并显示前DISPLAY个频谱图
    img = spectrogram_from_eeg(f'{PATH2}{eeg_id}.parquet', i < DISPLAY)
    all_eegs[eeg_id] = img  # 将频谱图存储在字典中，键为EEG ID


# 定义一个类，继承自PyTorch的nn.Module
class HMSHBACSpecModel(nn.Module):
    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            # checkpoint_path: str,
            in_channels: int,
            num_classes: int,
    ):
        # 调用父类的构造函数
        super().__init__()

        # 创建一个预先训练好的模型，根据模型名称、预训练标志、输入通道数和分类数进行初始化
        self.model = timm.create_model(model_name=model_name,
                                       pretrained=pretrained, pretrained_cfg_overlay=pretrained_cfg_overlay,
                                       num_classes=num_classes, in_chans=in_channels,
                                       )

    # 前向传播方法，用于处理输入数据x
    def forward(self, x):
        # 确保输入数据为半精度类型
        # x = x.half()  # 转换数据类型为half
        # 将输入数据传递给模型
        h = self.model(x)
        # 返回模型预测结果
        return h


# zfg 下面为拼接（原始+eeg to spec）

FilePath = tp.Union[str, Path]
EegIds = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]


class HMSHBACSpecDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            eeg_ids: tp.Sequence[EegIds],
            image_paths: tp.Sequence[FilePath],
            labels: tp.Sequence[Label],
            transform: A.Compose,
    ):
        self.eeg_ids = eeg_ids
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        eeg_id = self.eeg_ids[index]

        label = self.labels[index]
        img = np.load(img_path)
        eeg_spec = all_eegs[eeg_id]  # shape: (128, 256, 4)
        # eeg_spec 从 (128, 256, 4) 变为 512*256
        x1 = [eeg_spec[:, :, i:i + 1] for i in range(4)]
        result_img1 = np.zeros((512, 256), dtype=img.dtype)
        result_img1[:128, :] = x1[0].squeeze()
        result_img1[128:128 * 2, :] = x1[1].squeeze()
        result_img1[128 * 2:128 * 3, :] = x1[2].squeeze()
        result_img1[128 * 3:128 * 4, :] = x1[3].squeeze()
        eeg_spec = result_img1
        # eeg_spec 从 (128, 256, 4) 变为 512*256  结束
        # log transform
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        # normalize per image
        eps = 1e-6
        img_mean = img.mean(axis=(0, 1))
        img = img - img_mean
        img_std = img.std(axis=(0, 1))
        img = img / (img_std + eps)  # 400-300
        # 1、400*300 - 512*300 (4*128=4*(100+28)=4*(14+100+14))
        img1 = img[:100, :]
        img2 = img[100:200, :]
        img3 = img[200:300, :]
        img4 = img[300:400, :]

        result_img = np.zeros((512, 300), dtype=img.dtype)
        result_img[14:100 + 14, :] = img1
        result_img[100 + 14 * 3:200 + 14 * 3, :] = img2
        result_img[200 + 14 * 5:300 + 14 * 5, :] = img3
        result_img[300 + 14 * 7:400 + 14 * 7, :] = img4
        img = result_img
        # 2、512*300 -- 512*258
        img = img[:, 22:-22]
        result_img2 = np.zeros((512, 512), dtype=img.dtype)
        result_img2[:, :256] = img
        result_img2[:, 256:] = eeg_spec
        img = result_img2
        img = img[..., None]  # shape: (Hz, Time) -> (Hz, Time, Channel)
        img = self._apply_transform(img)

        return {"data": img, "target": label}

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img


class CFG:
    model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    img_size = 512
    max_epoch = 9
    batch_size = 32
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience = 5
    seed = 1086
    deterministic = True
    enable_amp = True
    device = "cuda"


def to_device(
        tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
        device: torch.device, *args, **kwargs
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {
            k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


def get_test_path_label(test: pd.DataFrame):
    """Get file path and dummy target info."""

    img_paths = []
    eeg_ids = []
    labels = np.full((len(test), 6), -1, dtype="float32")
    for spec_id in test["spectrogram_id"].values:
        img_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
        img_paths.append(img_path)

        test1 = test[test['spectrogram_id'] == spec_id]
        eeg_id = test1['eeg_id'].values[0]
        eeg_ids.append(eeg_id)
    test_data = {
        "eeg_ids": eeg_ids,
        "image_paths": img_paths,
        "labels": [l for l in labels]}

    return test_data


def get_test_transforms(CFG):
    test_transform = A.Compose([
        A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
        ToTensorV2(p=1.0)
    ])
    return test_transform


def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data"], device)
            y = model(x)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())

    pred_arr = np.concatenate(pred_list)
    del pred_list
    return pred_arr


test_preds_arr = np.zeros((N_FOLDS, len(test), N_CLASSES))
test_path_label = get_test_path_label(test)
test_transform = get_test_transforms(CFG)
test_dataset = HMSHBACSpecDataset(**test_path_label, transform=test_transform)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=CFG.batch_size, num_workers=4, shuffle=False, drop_last=False)

device = torch.device(CFG.device)

for fold_id in range(N_FOLDS):
    print(f"\n[fold {fold_id}]")

    # # get model
    model_path = Path(
        f"../hms-harmful-brain-activity-classification/output/stage2/best_model_fold{fold_id}.pth")
    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # # inference
    test_pred = run_inference_loop(model, test_loader, device)
    test_preds_arr[fold_id] = test_pred

    model_path = Path(
        f"../hms-harmful-brain-activity-classification/output/stage2/best_model_fold{fold_id}.pth")
    # model = HMSHBACSpecModel(
    #     model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=1)
    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    #     num+=1
    # # inference
    test_pred = run_inference_loop(model, test_loader, device)
    test_preds_arr[fold_id] = (test_preds_arr[fold_id] + test_pred) / 2.0

    del model
    torch.cuda.empty_cache()
    gc.collect()

test_pred = test_preds_arr.mean(axis=0)

test_pred_df = pd.DataFrame(
    test_pred, columns=CLASSES
)

test_pred_df = pd.concat([test[["eeg_id"]], test_pred_df], axis=1)

smpl_sub = pd.read_csv(DATA / "sample_submission.csv")

sub1 = pd.merge(
    smpl_sub[["eeg_id"]], test_pred_df, on="eeg_id", how="left")

# sub.to_csv("submission.csv", index=False)

sub1.head()

# zfg model3 + model4 + model5


# READ ALL SPECTROGRAMS
PATH2 = '../hms-harmful-brain-activity-classification/input/train-raw-csv/test_spectrograms/'
files2 = os.listdir(PATH2)
print(f'There are {len(files2)} test spectrogram parquets')

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 100 == 0: print(i, ', ', end='')
    tmp = pd.read_parquet(f'{PATH2}{f}')
    name = int(f.split('.')[0])
    spectrograms2[name] = tmp.iloc[:, 1:].values

# RENAME FOR DATALOADER
test['spec_id'] = test['spectrogram_id']

eeg_features = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4",
                "T6", "O2", "EKG"]

feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}

USE_WAVELET = None

NAMES = ['LL', 'LP', 'RP', 'RR']

FEATS = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
         ['Fp1', 'F3', 'C3', 'P3', 'O1'],
         ['Fp2', 'F8', 'T4', 'T6', 'O2'],
         ['Fp2', 'F4', 'C4', 'P4', 'O2']]


# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet='haar', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret = pywt.waverec(coeff, wavelet, mode='per')

    return ret


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)
    data = eeg.values

    nperseg = 150  # Length of each segment
    noverlap = 128  # Overlap between segments
    NFFT = max(256, 2 ** int(np.ceil(np.log2(nperseg))))

    freqs, t, spectrum_LL1 = signal.spectrogram(data[:, feature_to_index['Fp1']] - data[:, feature_to_index['F7']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL2 = signal.spectrogram(data[:, feature_to_index['F7']] - data[:, feature_to_index['T3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL3 = signal.spectrogram(data[:, feature_to_index['T3']] - data[:, feature_to_index['T5']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL4 = signal.spectrogram(data[:, feature_to_index['T5']] - data[:, feature_to_index['O1']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    LL = (spectrum_LL1 + spectrum_LL2 + spectrum_LL3 + spectrum_LL4) / 4

    freqs, t, spectrum_LP1 = signal.spectrogram(data[:, feature_to_index['Fp1']] - data[:, feature_to_index['F3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP2 = signal.spectrogram(data[:, feature_to_index['F3']] - data[:, feature_to_index['C3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP3 = signal.spectrogram(data[:, feature_to_index['C3']] - data[:, feature_to_index['P3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP4 = signal.spectrogram(data[:, feature_to_index['P3']] - data[:, feature_to_index['O1']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    LP = (spectrum_LP1 + spectrum_LP2 + spectrum_LP3 + spectrum_LP4) / 4

    freqs, t, spectrum_RP1 = signal.spectrogram(data[:, feature_to_index['Fp2']] - data[:, feature_to_index['F4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP2 = signal.spectrogram(data[:, feature_to_index['F4']] - data[:, feature_to_index['C4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP3 = signal.spectrogram(data[:, feature_to_index['C4']] - data[:, feature_to_index['P4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP4 = signal.spectrogram(data[:, feature_to_index['P4']] - data[:, feature_to_index['O2']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    RP = (spectrum_RP1 + spectrum_RP2 + spectrum_RP3 + spectrum_RP4) / 4

    freqs, t, spectrum_RL1 = signal.spectrogram(data[:, feature_to_index['Fp2']] - data[:, feature_to_index['F8']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RL2 = signal.spectrogram(data[:, feature_to_index['F8']] - data[:, feature_to_index['T4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RL3 = signal.spectrogram(data[:, feature_to_index['T4']] - data[:, feature_to_index['T6']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RL4 = signal.spectrogram(data[:, feature_to_index['T6']] - data[:, feature_to_index['O2']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    RL = (spectrum_RL1 + spectrum_RL2 + spectrum_RL3 + spectrum_RL4) / 4
    spectogram = np.concatenate((LL, LP, RP, RL), axis=0)
    #     all_eeg2spec[k]=spectogram

    return spectogram


# READ ALL EEG SPECTROGRAMS
PATH2 = '../hms-harmful-brain-activity-classification/input/train-raw-csv/test_eegs/'
DISPLAY = 1
EEG_IDS2 = test.eeg_id.unique()
all_eegs = {}

print('Converting Test EEG to Spectrograms...');
print()
for i, eeg_id in enumerate(EEG_IDS2):
    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f'{PATH2}{eeg_id}.parquet', i < DISPLAY)
    all_eegs[eeg_id] = img


class HMSHBACSpecModel(nn.Module):

    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            in_channels: int,
            num_classes: int,
    ):
        super().__init__()
        # self.model = timm.create_model(
        #     model_name=model_name, pretrained=pretrained,
        #     num_classes=num_classes, in_chans=in_channels)
        self.model = timm.create_model(model_name=model_name,
                                       pretrained=pretrained, pretrained_cfg_overlay=pretrained_cfg_overlay,
                                       num_classes=num_classes, in_chans=in_channels,
                                       )

    def forward(self, x):
        h = self.model(x)

        return h


FilePath = tp.Union[str, Path]
EegIds = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]


class HMSHBACSpecDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            eeg_ids: tp.Sequence[EegIds],
            image_paths: tp.Sequence[FilePath],
            labels: tp.Sequence[Label],
            transform: A.Compose,
    ):
        self.eeg_ids = eeg_ids
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        eeg_id = self.eeg_ids[index]

        label = self.labels[index]

        data = all_eegs[eeg_id]  #

        # Fill missing values with the specified constant
        mask = np.isnan(data)
        data[mask] = -1
        data = np.clip(data, np.exp(-6), np.exp(10))
        data = np.log(data)
        data_mean = data.mean(axis=(0, 1))
        data = data - data_mean
        img_std = data.std(axis=(0, 1))
        eps = 1e-6
        data = data / (img_std + eps)  #

        img = self._apply_transform(data)

        return {"data": img, "target": label}

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img


class CFG:
    model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    img_size = 512
    max_epoch = 9
    batch_size = 8
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience = 5
    seed = 1086
    deterministic = True
    enable_amp = True
    device = "cuda"


def to_device(
        tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
        device: torch.device, *args, **kwargs
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {
            k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


def get_test_path_label(test: pd.DataFrame):
    """Get file path and dummy target info."""

    img_paths = []
    eeg_ids = []
    labels = np.full((len(test), 6), -1, dtype="float32")
    for spec_id in test["spectrogram_id"].values:
        img_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
        img_paths.append(img_path)

        test1 = test[test['spectrogram_id'] == spec_id]
        eeg_id = test1['eeg_id'].values[0]
        eeg_ids.append(eeg_id)
    test_data = {
        "eeg_ids": eeg_ids,
        "image_paths": img_paths,
        "labels": [l for l in labels]}

    return test_data


def get_test_transforms(CFG):
    test_transform = A.Compose([
        A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
        ToTensorV2(p=1.0)
    ])
    return test_transform


def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data"], device)
            y = model(x)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())

    pred_arr = np.concatenate(pred_list)
    del pred_list
    return pred_arr


test_preds_arr = np.zeros((N_FOLDS, len(test), N_CLASSES))

test_path_label = get_test_path_label(test)
test_transform = get_test_transforms(CFG)
test_dataset = HMSHBACSpecDataset(**test_path_label, transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=CFG.batch_size, num_workers=4, shuffle=False, drop_last=False)

device = torch.device(CFG.device)

for fold_id in range(N_FOLDS):
    print(f"\n[fold {fold_id}]")
    num = 0
    # # get model
    model_path = Path(
        f"../hms-harmful-brain-activity-classification/output/stage2/best_model_fold{fold_id}.pth")
    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # # inference
    test_pred = run_inference_loop(model, test_loader, device)
    test_preds_arr[fold_id] = test_pred

    # # get model
    model_path = Path(
        f"../hms-harmful-brain-activity-classification/output/stage2/best_model_fold{fold_id}.pth")
    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # # inference
    test_pred = run_inference_loop(model, test_loader, device)
    test_preds_arr[fold_id] = (test_preds_arr[fold_id] + test_pred)

    model_path = Path(
        f"../hms-harmful-brain-activity-classification/output/stage2/best_model_fold{fold_id}.pth")

    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=1)

    model.load_state_dict(torch.load(model_path, map_location=device))

    # # inference
    test_pred = run_inference_loop(model, test_loader, device)
    test_preds_arr[fold_id] = (test_preds_arr[fold_id] + test_pred)
    #     num+=1
    del model
    torch.cuda.empty_cache()
    gc.collect()

test_pred = test_preds_arr.mean(axis=0)

test_pred_df = pd.DataFrame(
    test_pred, columns=CLASSES
)

test_pred_df = pd.concat([test[["eeg_id"]], test_pred_df], axis=1)

smpl_sub = pd.read_csv(DATA / "sample_submission.csv")

sub2 = pd.merge(
    smpl_sub[["eeg_id"]], test_pred_df, on="eeg_id", how="left")

# sub.to_csv("submission.csv", index=False)

sub2.head()

# zfg model6
# READ ALL SPECTROGRAMS
PATH2 = '../hms-harmful-brain-activity-classification/input/train-raw-csv/test_spectrograms/'
files2 = os.listdir(PATH2)
print(f'There are {len(files2)} test spectrogram parquets')

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 100 == 0: print(i, ', ', end='')
    tmp = pd.read_parquet(f'{PATH2}{f}')
    name = int(f.split('.')[0])
    spectrograms2[name] = tmp.iloc[:, 1:].values

# RENAME FOR DATALOADER
test['spec_id'] = test['spectrogram_id']
# test = test.rename({'spectrogram_id':'spec_id'},axis=1)


eeg_features = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4",
                "T6", "O2", "EKG"]

feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}

USE_WAVELET = None

NAMES = ['LL', 'LP', 'RP', 'RR']

FEATS = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
         ['Fp1', 'F3', 'C3', 'P3', 'O1'],
         ['Fp2', 'F8', 'T4', 'T6', 'O2'],
         ['Fp2', 'F4', 'C4', 'P4', 'O2']]


# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet='haar', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret = pywt.waverec(coeff, wavelet, mode='per')

    return ret


def spectrogram_from_eeg(parquet_path, display=False, type=1):
    eeg = pd.read_parquet(parquet_path)
    data = eeg.values
    if type == 2:
        data = data[4000:6000, :]

    nperseg = 150  # Length of each segment
    noverlap = 128  # Overlap between segments
    NFFT = max(256, 2 ** int(np.ceil(np.log2(nperseg))))

    freqs, t, spectrum_LL1 = signal.spectrogram(data[:, feature_to_index['Fp1']] - data[:, feature_to_index['F7']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL2 = signal.spectrogram(data[:, feature_to_index['F7']] - data[:, feature_to_index['T3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL3 = signal.spectrogram(data[:, feature_to_index['T3']] - data[:, feature_to_index['T5']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL4 = signal.spectrogram(data[:, feature_to_index['T5']] - data[:, feature_to_index['O1']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    LL = (spectrum_LL1 + spectrum_LL2 + spectrum_LL3 + spectrum_LL4) / 4

    freqs, t, spectrum_LP1 = signal.spectrogram(data[:, feature_to_index['Fp1']] - data[:, feature_to_index['F3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP2 = signal.spectrogram(data[:, feature_to_index['F3']] - data[:, feature_to_index['C3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP3 = signal.spectrogram(data[:, feature_to_index['C3']] - data[:, feature_to_index['P3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP4 = signal.spectrogram(data[:, feature_to_index['P3']] - data[:, feature_to_index['O1']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    LP = (spectrum_LP1 + spectrum_LP2 + spectrum_LP3 + spectrum_LP4) / 4

    freqs, t, spectrum_RP1 = signal.spectrogram(data[:, feature_to_index['Fp2']] - data[:, feature_to_index['F4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP2 = signal.spectrogram(data[:, feature_to_index['F4']] - data[:, feature_to_index['C4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP3 = signal.spectrogram(data[:, feature_to_index['C4']] - data[:, feature_to_index['P4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP4 = signal.spectrogram(data[:, feature_to_index['P4']] - data[:, feature_to_index['O2']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    RP = (spectrum_RP1 + spectrum_RP2 + spectrum_RP3 + spectrum_RP4) / 4

    freqs, t, spectrum_RL1 = signal.spectrogram(data[:, feature_to_index['Fp2']] - data[:, feature_to_index['F8']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RL2 = signal.spectrogram(data[:, feature_to_index['F8']] - data[:, feature_to_index['T4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RL3 = signal.spectrogram(data[:, feature_to_index['T4']] - data[:, feature_to_index['T6']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RL4 = signal.spectrogram(data[:, feature_to_index['T6']] - data[:, feature_to_index['O2']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    RL = (spectrum_RL1 + spectrum_RL2 + spectrum_RL3 + spectrum_RL4) / 4
    spectogram = np.concatenate((LL, LP, RP, RL), axis=0)
    #     all_eeg2spec[k]=spectogram

    return spectogram


# READ ALL EEG SPECTROGRAMS
PATH2 = '../hms-harmful-brain-activity-classification/input/train-raw-csv/test_eegs/'
DISPLAY = 1
EEG_IDS2 = test.eeg_id.unique()
all_eegs = {}

print('Converting Test EEG to Spectrograms...');
print()
for i, eeg_id in enumerate(EEG_IDS2):
    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f'{PATH2}{eeg_id}.parquet', i < DISPLAY, type=1)
    all_eegs[eeg_id] = img

all_eegs1 = {}
for i, eeg_id in enumerate(EEG_IDS2):
    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f'{PATH2}{eeg_id}.parquet', i < DISPLAY, type=2)
    all_eegs1[eeg_id] = img


class HMSHBACSpecModel(nn.Module):

    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            in_channels: int,
            num_classes: int,
    ):
        super().__init__()
        # self.model = timm.create_model(
        #     model_name=model_name, pretrained=pretrained,
        #     num_classes=num_classes, in_chans=in_channels)

        self.model = timm.create_model(model_name=model_name,
                                       pretrained=pretrained, pretrained_cfg_overlay=pretrained_cfg_overlay,
                                       num_classes=num_classes, in_chans=in_channels,
                                       )

    def forward(self, x):
        h = self.model(x)

        return h


FilePath = tp.Union[str, Path]
EegIds = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]


class HMSHBACSpecDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            eeg_ids: tp.Sequence[EegIds],
            image_paths: tp.Sequence[FilePath],
            labels: tp.Sequence[Label],
            transform: A.Compose,
    ):
        self.eeg_ids = eeg_ids
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        eeg_id = self.eeg_ids[index]

        #         print(f"img_path:{img_path},eeg_id:{eeg_id}")

        label = self.labels[index]

        #  (516, 448)
        data = all_eegs[eeg_id]
        # Fill missing values with the specified constant
        mask = np.isnan(data)
        data[mask] = -1
        data = np.clip(data, np.exp(-6), np.exp(10))
        data = np.log(data)
        data_mean = data.mean(axis=(0, 1))
        data = data - data_mean
        img_std = data.std(axis=(0, 1))
        eps = 1e-6
        data = data / (img_std + eps)  # 400-300

        # (516, 85)
        data1 = all_eegs1[eeg_id]
        # Fill missing values with the specified constant
        mask1 = np.isnan(data1)
        data1[mask1] = -1
        data1 = np.clip(data1, np.exp(-6), np.exp(10))
        data1 = np.log(data1)
        data_mean1 = data1.mean(axis=(0, 1))
        data1 = data1 - data_mean1
        img_std1 = data1.std(axis=(0, 1))
        eps = 1e-6
        data1 = data1 / (img_std1 + eps)  # 400-300

        result_img = np.zeros((516, 533), dtype=data1.dtype)
        result_img[:, :448] = data
        result_img[:, 448:] = data1
        result_img = result_img[..., None]

        img = self._apply_transform(result_img)

        return {"data": img, "target": label}

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img


class CFG:
    model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    img_size = 512
    max_epoch = 9
    batch_size = 32
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience = 5
    seed = 1086
    deterministic = True
    enable_amp = True
    device = "cuda"


def to_device(
        tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
        device: torch.device, *args, **kwargs
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {
            k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


def get_test_path_label(test: pd.DataFrame):
    """Get file path and dummy target info."""

    img_paths = []
    eeg_ids = []
    labels = np.full((len(test), 6), -1, dtype="float32")
    for spec_id in test["spectrogram_id"].values:
        img_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
        img_paths.append(img_path)

        test1 = test[test['spectrogram_id'] == spec_id]
        eeg_id = test1['eeg_id'].values[0]
        eeg_ids.append(eeg_id)
    test_data = {
        "eeg_ids": eeg_ids,
        "image_paths": img_paths,
        "labels": [l for l in labels]}

    return test_data


def get_test_transforms(CFG):
    test_transform = A.Compose([
        A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
        ToTensorV2(p=1.0)
    ])
    return test_transform


def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data"], device)
            y = model(x)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())

    pred_arr = np.concatenate(pred_list)
    del pred_list
    return pred_arr


test_preds_arr = np.zeros((N_FOLDS, len(test), N_CLASSES))

test_path_label = get_test_path_label(test)
test_transform = get_test_transforms(CFG)
test_dataset = HMSHBACSpecDataset(**test_path_label, transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=CFG.batch_size, num_workers=4, shuffle=False, drop_last=False)

device = torch.device(CFG.device)

for fold_id in range(N_FOLDS):
    print(f"\n[fold {fold_id}]")
    num = 0

    # # get model
    if fold_id == 2:
        model_path = Path(
            f"../hms-harmful-brain-activity-classification/output/stage2/best_model_fold{fold_id}.pth")
    else:
        model_path = Path(
            f"../hms-harmful-brain-activity-classification/output/stage2/best_model_fold{fold_id}.pth")
    # model = HMSHBACSpecModel(
    #     model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=1)
    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # # inference
    test_pred = run_inference_loop(model, test_loader, device)
    test_preds_arr[fold_id] = test_pred

    del model
    torch.cuda.empty_cache()
    gc.collect()

test_pred = test_preds_arr.mean(axis=0)

test_pred_df = pd.DataFrame(
    test_pred, columns=CLASSES
)

test_pred_df = pd.concat([test[["eeg_id"]], test_pred_df], axis=1)

smpl_sub = pd.read_csv(DATA / "sample_submission.csv")

sub4 = pd.merge(
    smpl_sub[["eeg_id"]], test_pred_df, on="eeg_id", how="left")

# sub.to_csv("submission.csv", index=False)

sub4.head()

# zfg model7


import gc
import math
import numpy as np
import pandas as pd

from glob import glob
from typing import Dict, Union
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.append('../kaggle_kl_div')

import warnings

warnings.filterwarnings("ignore")


class CFG:
    VERSION = 88
    model_name = "resnet1d_gru"

    seed = 2024
    batch_size = 32
    num_workers = 0

    fixed_kernel_size = 5

    kernels = [3, 5, 7, 9, 11]

    linear_layer_features = 304  # 1/5  Signal = 2_000

    seq_length = 50  # Second's
    sampling_rate = 200  # Hz
    nsamples = seq_length * sampling_rate  # Число семплов
    out_samples = nsamples // 5

    freq_channels = []  # [(8.0, 12.0)]; [(0.5, 4.5)]
    filter_order = 2
    random_close_zone = 0.0  # 0.2

    target_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
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

    feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}
    simple_features = []  # 'Fz', 'Cz', 'Pz', 'EKG'

    n_map_features = len(map_features)
    in_channels = n_map_features + n_map_features * len(freq_channels) + len(simple_features)
    target_size = len(target_cols)

    PATH = "../hms-harmful-brain-activity-classification/input/train-raw-csv/"
    test_eeg = "../hms-harmful-brain-activity-classification/input/train-raw-csv/test_eegs/"
    test_csv = "../hms-harmful-brain-activity-classification/input/train-raw-csv/test.csv"


koef_1 = 1.0
model_weights = [
    {
        'bandpass_filter': {'low': 0.5, 'high': 20, 'order': 2},
        'file_data':
            [
                {'koef': koef_1,
                 'file_mask': "../hms-harmful-brain-activity-classification/output_7/pop_2_weight_oof/*_best.pth"},
            ]
    },
]


def init_logger(log_file="./test.log"):
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
        self.bandpass_filter = bandpass_filter
        self.rand_filter = rand_filter

    def __len__(self):
        """
        Length of dataset.
        """
        # Обозначает количество пакетов за эпоху
        return len(self.df)

    def __getitem__(self, index):
        """
        Get one item.
        """
        # Сгенерировать один пакет данных
        X, y_prob = self.__data_generation(index)
        if self.downsample is not None:
            X = X[:: self.downsample, :]
        output = {
            "eeg": torch.tensor(X, dtype=torch.float32),
            "labels": torch.tensor(y_prob, dtype=torch.float32),
        }
        return output

    def __data_generation(self, index):
        # Генерирует данные, содержащие образцы размера партии
        X = np.zeros(
            (CFG.out_samples, CFG.in_channels), dtype="float32"
        )  # Size=(10000, 14)

        row = self.df.iloc[index]  # Строка Pandas
        data = self.eegs[row.eeg_id]  # Size=(10000, 8)
        if CFG.nsamples != CFG.out_samples:
            if self.mode != "train":
                offset = (CFG.nsamples - CFG.out_samples) // 2
            else:
                # offset = random.randint(0, CFG.nsamples - CFG.out_samples)
                offset = ((CFG.nsamples - CFG.out_samples) * random.randint(0, 1000)) // 1000
            data = data[offset:offset + CFG.out_samples, :]

        for i, (feat_a, feat_b) in enumerate(CFG.map_features):
            if self.mode == "train" and CFG.random_close_zone > 0 and random.uniform(0.0, 1.0) <= CFG.random_close_zone:
                continue

            diff_feat = (
                    data[:, CFG.feature_to_index[feat_a]]
                    - data[:, CFG.feature_to_index[feat_b]]
            )  # Size=(10000,)

            if not self.bandpass_filter is None:
                diff_feat = butter_bandpass_filter(
                    diff_feat,
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

        # Обрезать края превышающие значения [-1024, 1024]
        X = np.clip(X, -1024, 1024)

        # Замените NaN нулем и разделить все на 32
        X = np.nan_to_num(X, nan=0) / 32.0

        # обрезать полосовым фильтром верхнюю границу в 20 Hz.
        X = butter_lowpass_filter(X, order=CFG.filter_order)  # 4

        y_prob = np.zeros(CFG.target_size, dtype="float32")  # Size=(6,)
        if self.mode != "test":
            y_prob = row[CFG.target_cols].values.astype(np.float32)

        return X, y_prob


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
        new_rnn_h = rnn_out[:, -1, :]  # <~~

        new_out = torch.cat([out, new_rnn_h], dim=1)
        return new_out

    def forward(self, x):
        new_out = self.extract_features(x)
        result = self.fc(new_out)
        return result


def inference_function(test_loader, model, device):
    model.eval()  # set model in evaluation mode
    softmax = nn.Softmax(dim=1)
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, batch in enumerate(tqdm_test_loader):
            X = batch.pop("eeg").to(device)  # send inputs to `device`
            batch_size = X.size(0)
            with torch.no_grad():
                y_preds = model(X)  # forward propagation pass
            y_preds = softmax(y_preds)
            preds.append(y_preds.to("cpu").numpy())  # save predictions

    prediction_dict["predictions"] = np.concatenate(
        preds
    )  # np.array() of shape (fold_size, target_cols)
    return prediction_dict


test_df = pd.read_csv(CFG.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()

test_eeg_parquet_paths = glob(CFG.test_eeg + "*.parquet")
test_eeg_df = pd.read_parquet(test_eeg_parquet_paths[0])
test_eeg_features = test_eeg_df.columns
print(f"There are {len(test_eeg_features)} raw eeg features")
print(list(test_eeg_features))
del test_eeg_df
_ = gc.collect()

all_eegs = {}
eeg_ids = test_df.eeg_id.unique()
for i, eeg_id in tqdm(enumerate(eeg_ids)):
    # Save EEG to Python dictionary of numpy arrays
    eeg_path = CFG.test_eeg + str(eeg_id) + ".parquet"
    data = eeg_from_parquet(eeg_path)
    all_eegs[eeg_id] = data

koef_sum = 0
koef_count = 0
predictions = []
files = []

for model_block in model_weights:
    test_dataset = EEGDataset(
        df=test_df,
        batch_size=CFG.batch_size,
        mode="test",
        eegs=all_eegs,
        bandpass_filter=model_block['bandpass_filter']
    )

    if len(predictions) == 0:
        output = test_dataset[0]
        X = output["eeg"]
        print(f"X shape: {X.shape}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = EEGNet(
        kernels=CFG.kernels,
        in_channels=CFG.in_channels,
        fixed_kernel_size=CFG.fixed_kernel_size,
        num_classes=CFG.target_size,
        linear_layer_features=CFG.linear_layer_features,
    )

    for file_line in model_block['file_data']:
        koef = file_line['koef']
        for weight_model_file in glob(file_line['file_mask']):
            files.append(weight_model_file)
            checkpoint = torch.load(weight_model_file, map_location=device)
            model.load_state_dict(checkpoint["model"])
            model.to(device)
            prediction_dict = inference_function(test_loader, model, device)
            predict = prediction_dict["predictions"]
            predict *= koef
            koef_sum += koef
            koef_count += 1
            predictions.append(predict)
            torch.cuda.empty_cache()
            gc.collect()

predictions = np.array(predictions)
koef_sum /= koef_count
predictions /= koef_sum
predictions = np.mean(predictions, axis=0)

sub5 = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub5[CFG.target_cols] = predictions

# sub.to_csv(f"submission.csv", index=False)
print(f"Submission shape: {sub5.shape}")
sub5.head()

# zfg model8_1


import os
import gc
import pywt
import time
import torch
import librosa
import random

import pandas as pd
import scipy.ndimage as ndi
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy import signal

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')

pretrained_cfg_overlay = {
    'file': r"/home/laplace/.cache/huggingface/hub/models--timm--tf_efficientnet_b0.ns_jft_in1k/pytorch_model.bin"}
df = pd.read_csv('../hms-harmful-brain-activity-classification/input_1D_2D/test.csv')
label_cols = df.columns[-6:]


# zfg configuration

class config:
    BATCH_SIZE = 1
    backbone_2d = 'tf_efficientnet_b0.ns_jft_in1k'
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False
    num_channels = 8
    pretrained = False
    augment = False
    image_transform = transforms.Resize((512, 512))


class paths:
    OUTPUT_DIR = "../hms-harmful-brain-activity-classification/output_1D_2D/"
    TEST_CSV = "../hms-harmful-brain-activity-classification/input_1D_2D/test.csv"
    TEST_EEGS = "../hms-harmful-brain-activity-classification/input_1D_2D/test_eegs/"
    TEST_SPECTROGRAMS = "../hms-harmful-brain-activity-classification/input_1D_2D/test_spectrograms/"


model_weights = [f"../hms-harmful-brain-activity-classification/output_1D_2D/wavenet_fold_{x}_s2_best_val.pth" for x in
                 range(5)]

# zfg utils

USE_WAVELET = None

NAMES = ['LL', 'LP', 'RP', 'RR']

FEATS = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
         ['Fp1', 'F3', 'C3', 'P3', 'O1'],
         ['Fp2', 'F8', 'T4', 'T6', 'O2'],
         ['Fp2', 'F4', 'C4', 'P4', 'O2']]


def maddest(d, axis: int = None):
    """
    Denoise function.
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x: np.ndarray, wavelet: str = 'haar', level: int = 1):
    coeff = pywt.wavedec(x, wavelet, mode="per")  # multilevel 1D Discrete Wavelet Transform of data.
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    output = pywt.waverec(coeff, wavelet, mode='per')
    return output


def spectrogram_from_eeg(parquet_path, display=False):
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle:middle + 10_000]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128, 256, 4), dtype='float32')

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):

            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            # DENOISE
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x) // 256,
                                                      n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect='auto', origin='lower')
            plt.title(f'EEG {eeg_id} - Spectrogram {NAMES[k]}')

    if display:
        plt.show()
        plt.figure(figsize=(10, 5))
        offset = 0
        for k in range(4):
            if k > 0: offset -= signals[3 - k].min()
            plt.plot(range(10_000), signals[k] + offset, label=NAMES[3 - k])
            offset += signals[3 - k].max()
        plt.legend()
        plt.title(f'EEG {eeg_id} Signals')
        plt.show()
        print();
        print('#' * 25);
        print()

    return img


def plot_spectrogram(spectrogram_path: str):
    sample_spect = pd.read_parquet(spectrogram_path)

    split_spect = {
        "LL": sample_spect.filter(regex='^LL', axis=1),
        "RL": sample_spect.filter(regex='^RL', axis=1),
        "RP": sample_spect.filter(regex='^RP', axis=1),
        "LP": sample_spect.filter(regex='^LP', axis=1),
    }

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    axes = axes.flatten()
    label_interval = 5
    for i, split_name in enumerate(split_spect.keys()):
        ax = axes[i]
        img = ax.imshow(np.log(split_spect[split_name]).T, cmap='viridis', aspect='auto', origin='lower')
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Log(Value)')
        ax.set_title(split_name)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time")

        ax.set_yticks(np.arange(len(split_spect[split_name].columns)))
        ax.set_yticklabels([column_name[3:] for column_name in split_spect[split_name].columns])
        frequencies = [column_name[3:] for column_name in split_spect[split_name].columns]
        ax.set_yticks(np.arange(0, len(split_spect[split_name].columns), label_interval))
        ax.set_yticklabels(frequencies[::label_interval])
    plt.tight_layout()
    plt.show()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def sep():
    print("-" * 100)


def create_spectrogram_with_cusignal(eeg_data, eeg_id, start, duration=50,
                                     low_cut_freq=0.7, high_cut_freq=20, order_band=5,
                                     spec_size_freq=267, spec_size_time=30,
                                     nperseg_=1500, noverlap_=1483, nfft_=2750,
                                     sigma_gaussian=0.7,
                                     mean_montage_names=4):
    electrode_names = ['LL', 'RL', 'LP', 'RP']

    electrode_pairs = [
        ['Fp1', 'F7', 'T3', 'T5', 'O1'],
        ['Fp2', 'F8', 'T4', 'T6', 'O2'],
        ['Fp1', 'F3', 'C3', 'P3', 'O1'],
        ['Fp2', 'F4', 'C4', 'P4', 'O2']
    ]

    # Filter specifications
    nyquist_freq = 0.5 * 200
    low_cut_freq_normalized = low_cut_freq / nyquist_freq
    high_cut_freq_normalized = high_cut_freq / nyquist_freq

    # Bandpass and notch filter
    bandpass_coefficients = butter(order_band, [low_cut_freq_normalized, high_cut_freq_normalized], btype='band')
    notch_coefficients = iirnotch(w0=60, Q=30, fs=200)

    spec_size = duration * 200
    start = start * 200
    real_start = start + (10_000 // 2) - (spec_size // 2)
    eeg_data = eeg_data.iloc[real_start:real_start + spec_size]

    # Spectrogram parameters
    fs = 200
    nperseg = nperseg_
    noverlap = noverlap_
    nfft = nfft_

    if spec_size_freq <= 0 or spec_size_time <= 0:
        frequencias_size = int((nfft // 2) / 5.15198) + 1
        segmentos = int((spec_size - noverlap) / (nperseg - noverlap))
    else:
        frequencias_size = spec_size_freq
        segmentos = spec_size_time

    spectrogram = cp.zeros((frequencias_size, segmentos, 4), dtype='float32')

    processed_eeg = {}

    for i, name in enumerate(electrode_names):
        cols = electrode_pairs[i]
        processed_eeg[name] = np.zeros(spec_size)
        for j in range(4):
            # Compute differential signals
            # print(cols[j])
            signal = cp.array(eeg_data[cols[j]].values - eeg_data[cols[j + 1]].values)

            # Handle NaNs
            mean_signal = cp.nanmean(signal)
            signal = cp.nan_to_num(signal, nan=mean_signal) if cp.isnan(signal).mean() < 1 else cp.zeros_like(signal)

            # Filter bandpass and notch
            signal_filtered = filtfilt(*notch_coefficients, signal.get())
            signal_filtered = filtfilt(*bandpass_coefficients, signal_filtered)
            signal = cp.asarray(signal_filtered)

            frequencies, times, Sxx = cusignal.spectrogram(signal, fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

            # Filter frequency range
            valid_freqs = (frequencies >= 0.59) & (frequencies <= 20)
            frequencies_filtered = frequencies[valid_freqs]
            Sxx_filtered = Sxx[valid_freqs, :]

            # Logarithmic transformation and normalization using Cupy
            spectrogram_slice = cp.clip(Sxx_filtered, cp.exp(-4), cp.exp(6))
            spectrogram_slice = cp.log10(spectrogram_slice)

            normalization_epsilon = 1e-6
            mean = spectrogram_slice.mean(axis=(0, 1), keepdims=True)
            std = spectrogram_slice.std(axis=(0, 1), keepdims=True)
            spectrogram_slice = (spectrogram_slice - mean) / (std + normalization_epsilon)

            spectrogram[:, :, i] += spectrogram_slice
            processed_eeg[f'{cols[j]}_{cols[j + 1]}'] = signal.get()
            processed_eeg[name] += signal.get()

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        if mean_montage_names > 0:
            spectrogram[:, :, i] /= mean_montage_names

    # Convert to NumPy and apply Gaussian filter
    spectrogram_np = cp.asnumpy(spectrogram)
    if sigma_gaussian > 0.0:
        spectrogram_np = gaussian_filter(spectrogram_np, sigma=sigma_gaussian)

    # Filter EKG signal
    ekg_signal_filtered = filtfilt(*notch_coefficients, eeg_data["EKG"].values)
    ekg_signal_filtered = filtfilt(*bandpass_coefficients, ekg_signal_filtered)
    processed_eeg['EKG'] = np.array(ekg_signal_filtered)

    return spectrogram_np, processed_eeg


def create_spectogram_competition(spec_id, seconds_min):
    spec = pd.read_parquet(paths.TEST_SPECTROGRAMS + f'{spec_id}.parquet')
    #     spec = pd.read_parquet('/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/' + f'{spec_id}.parquet')

    inicio = (seconds_min) // 2
    img = spec.fillna(0).values[:, 1:].T.astype("float32")
    img = img[:, inicio:inicio + 300]

    # Log transform and normalize
    img = np.clip(img, np.exp(-4), np.exp(6))
    img = np.log(img)
    eps = 1e-6
    img_mean = img.mean()
    img_std = img.std()
    img = (img - img_mean) / (img_std + eps)

    return img


def create_512_spec(eeg_data, eeg_id, spec_id, start_second=0, seconds_min=0):
    image_50s, _ = create_spectrogram_with_cusignal(eeg_data=eeg_data, eeg_id=eeg_id, start=start_second, duration=50,
                                                    low_cut_freq=0.7, high_cut_freq=20, order_band=5,
                                                    spec_size_freq=267, spec_size_time=501,
                                                    nperseg_=1500, noverlap_=1483, nfft_=2750,
                                                    sigma_gaussian=0.0,
                                                    mean_montage_names=4)
    image_10s, _ = create_spectrogram_with_cusignal(eeg_data=eeg_data, eeg_id=eeg_id, start=start_second, duration=10,
                                                    low_cut_freq=0.7, high_cut_freq=20, order_band=5,
                                                    spec_size_freq=100, spec_size_time=291,
                                                    nperseg_=260, noverlap_=254, nfft_=1030,
                                                    sigma_gaussian=0.0,
                                                    mean_montage_names=4)
    image_10m = create_spectogram_competition(spec_id, seconds_min)

    imagem_final_unico_canal = np.zeros((1068, 501))
    for j in range(4):
        inicio = j * 267
        fim = inicio + 267
        imagem_final_unico_canal[inicio:fim, :] = image_50s[:, :, j]

    imagem_final_unico_canal2 = np.zeros((400, 291))
    for n in range(4):
        inicio = n * 100
        fim = inicio + 100
        imagem_final_unico_canal2[inicio:fim, :] = image_10s[:, :, n]

    imagem_final_unico_canal_resized = cv2.resize(imagem_final_unico_canal, (400, 800), interpolation=cv2.INTER_AREA)
    imagem_final_unico_canal2_resized = cv2.resize(imagem_final_unico_canal2, (300, 400), interpolation=cv2.INTER_AREA)
    eeg_new_resized = cv2.resize(image_10m, (300, 400), interpolation=cv2.INTER_AREA)
    imagem_final = np.zeros((800, 700), dtype=np.float32)
    imagem_final[0:800, 0:400] = imagem_final_unico_canal_resized
    imagem_final[0:400, 400:700] = imagem_final_unico_canal2_resized
    imagem_final[400:800, 400:700] = eeg_new_resized
    imagem_final = imagem_final[::-1]

    imagem_final = cv2.resize(imagem_final, (512, 512), interpolation=cv2.INTER_AREA)

    return imagem_final


def create_spectrogram(data):
    """This function will create the spectrograms based on the EEG data with the 'magic formula'."""
    nperseg = 150  # Length of each segment
    noverlap = 128  # Overlap between segments
    NFFT = max(256, 2 ** int(np.ceil(np.log2(nperseg))))

    freqs, t, spectrum_LL1 = signal.spectrogram(data[:, feature_to_index['Fp1']] - data[:, feature_to_index['F7']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL2 = signal.spectrogram(data[:, feature_to_index['F7']] - data[:, feature_to_index['T3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    freqs, t, spectrum_LL3 = signal.spectrogram(data[:, feature_to_index['T3']] - data[:, feature_to_index['T5']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    freqs, t, spectrum_LL4 = signal.spectrogram(data[:, feature_to_index['T5']] - data[:, feature_to_index['O1']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    LL = (spectrum_LL1 + spectrum_LL2 + spectrum_LL3 + spectrum_LL4) / 4

    freqs, t, spectrum_LP1 = signal.spectrogram(data[:, feature_to_index['Fp1']] - data[:, feature_to_index['F3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    freqs, t, spectrum_LP2 = signal.spectrogram(data[:, feature_to_index['F3']] - data[:, feature_to_index['C3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    freqs, t, spectrum_LP3 = signal.spectrogram(data[:, feature_to_index['C3']] - data[:, feature_to_index['P3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    freqs, t, spectrum_LP4 = signal.spectrogram(data[:, feature_to_index['P3']] - data[:, feature_to_index['O1']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    LP = (spectrum_LP1 + spectrum_LP2 + spectrum_LP3 + spectrum_LP4) / 4

    freqs, t, spectrum_RP1 = signal.spectrogram(data[:, feature_to_index['Fp2']] - data[:, feature_to_index['F4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    freqs, t, spectrum_RP2 = signal.spectrogram(data[:, feature_to_index['F4']] - data[:, feature_to_index['C4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    freqs, t, spectrum_RP3 = signal.spectrogram(data[:, feature_to_index['C4']] - data[:, feature_to_index['P4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    freqs, t, spectrum_RP4 = signal.spectrogram(data[:, feature_to_index['P4']] - data[:, feature_to_index['O2']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    RP = (spectrum_RP1 + spectrum_RP2 + spectrum_RP3 + spectrum_RP4) / 4

    freqs, t, spectrum_RL1 = signal.spectrogram(data[:, feature_to_index['Fp2']] - data[:, feature_to_index['F8']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    freqs, t, spectrum_RL2 = signal.spectrogram(data[:, feature_to_index['F8']] - data[:, feature_to_index['T4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    freqs, t, spectrum_RL3 = signal.spectrogram(data[:, feature_to_index['T4']] - data[:, feature_to_index['T6']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    freqs, t, spectrum_RL4 = signal.spectrogram(data[:, feature_to_index['T6']] - data[:, feature_to_index['O2']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)

    RL = (spectrum_RL1 + spectrum_RL2 + spectrum_RL3 + spectrum_RL4) / 4
    spectogram = np.concatenate((LL, LP, RP, RL), axis=0)
    return spectogram


label_to_num = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
num_to_label = {v: k for k, v in label_to_num.items()}
seed_everything(config.SEED)


def eeg_from_parquet(parquet_path: str) -> np.ndarray:
    # === Extract middle 50 seconds ===
    eeg = pd.read_parquet(parquet_path, columns=eeg_features)
    rows = len(eeg)
    offset = (rows - 10_000) // 2  # 50 * 200 = 10_000
    eeg = eeg.iloc[offset:offset + 10_000]  # middle 50 seconds, has the same amount of readings to left and right
    # === Convert to numpy ===
    data = np.zeros((10_000, len(eeg_features)))  # create placeholder of same shape with zeros
    for index, feature in enumerate(eeg_features):
        x = eeg[feature].values.astype('float32')  # convert to float32
        mean = np.nanmean(x)  # arithmetic mean along the specified axis, ignoring NaNs
        nan_percentage = np.isnan(x).mean()  # percentage of NaN values in feature
        # === Fill nan values ===
        if nan_percentage < 1:  # if some values are nan, but not all
            x = np.nan_to_num(x, nan=mean)
        else:  # if all values are nan
            x[:] = 0
        data[:, index] = x

    return data


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def sep():
    print("-" * 100)


target_preds = [x + "_pred" for x in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
label_to_num = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
num_to_label = {v: k for k, v in label_to_num.items()}
seed_everything(config.SEED)

# zfg load data

test_df = pd.read_csv(paths.TEST_CSV)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()

eeg_parquet_paths = glob(paths.TEST_EEGS + "*.parquet")
eeg_df = pd.read_parquet(eeg_parquet_paths[0])
eeg_features = eeg_df.columns
print(f'There are {len(eeg_features)} raw eeg features')
print(list(eeg_features))
# eeg_features = ['Fp1','T3','C3','O1','Fp2','C4','T4','O2']
feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}

CREATE_EEGS = False
all_eegs = {}
visualize = 1
eeg_paths = glob(paths.TEST_EEGS + "*.parquet")
eeg_ids = test_df.eeg_id.unique()

for i, eeg_id in tqdm(enumerate(eeg_ids)):
    # Save EEG to Python dictionary of numpy arrays
    eeg_path = paths.TEST_EEGS + str(eeg_id) + ".parquet"
    data = eeg_from_parquet(eeg_path)
    all_eegs[eeg_id] = data

# zfg read spectrograms

paths_spectrograms = glob(paths.TEST_SPECTROGRAMS + "*.parquet")
print(f'There are {len(paths_spectrograms)} spectrogram parquets')
all_spectrograms = {}

for file_path in tqdm(paths_spectrograms):
    aux = pd.read_parquet(file_path)
    name = int(file_path.split("/")[-1].split('.')[0])
    all_spectrograms[name] = aux.iloc[:, 1:].values
    del aux

if config.VISUALIZE:
    idx = np.random.randint(0, len(paths_spectrograms))
    spectrogram_path = paths_spectrograms[idx]
    plot_spectrogram(spectrogram_path)

# zfg read eeg spectrograms


paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
print(f'There are {len(paths_eegs)} EEG spectrograms')
all_eegs_specs = {}
counter = 0

for file_path in tqdm(paths_eegs):
    eeg_id = file_path.split("/")[-1].split(".")[0]
    eeg_spectrogram = spectrogram_from_eeg(file_path, counter < 1)
    all_eegs_specs[int(eeg_id)] = eeg_spectrogram
    counter += 1

all_eegs_512_spec = {}

for i in range(len(test_df)):
    row = test_df.iloc[i]
    eeg_id = row['eeg_id']
    eeg_data = pd.read_parquet(paths.TEST_EEGS + f'{eeg_id}.parquet')
    all_eegs_512_spec[str(eeg_id)] = create_512_spec(eeg_data, eeg_id, row['spectrogram_id'])

# unique
data_mean = 0.00011579196009429602
data_std = 4.5827806440634316

# zfg dataset

from scipy.signal import lfilter


def butter_lowpass_filter(data, cutoff_freq: int = 20, sampling_rate: int = 200, order: int = 4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


class CustomDataset(Dataset):
    def __init__(
            self, df, config, mode='train', specs=all_spectrograms, eeg_specs=all_eegs_specs, downsample=5,
            augment=config.augment, data_mean=data_mean, data_std=data_std
    ):
        self.df = df
        self.config = config
        self.mode = mode
        self.spectrograms = specs
        self.eeg_spectrograms = eeg_specs
        self.downsample = downsample
        self.augment = augment
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.all_eegs_512_spec = all_eegs_512_spec
        self.data_mean = data_mean
        self.data_std = data_std

    def __len__(self):
        """
        Length of dataset.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Get one item.
        """
        X1, X2, y = self.__data_generation(index)

        output = {
            "X1": torch.tensor(X1, dtype=torch.float32),
            "X2": torch.tensor(X2, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32)
        }
        return output

    def get_datasetwidenorm(self, eeg_id):
        """This function will get the batch and preprocess it."""
        # Set a small epsilon to avoid division by zero
        eps = 1e-6

        # Read data from parquet file
        data = all_eegs[eeg_id]
        data = create_spectrogram(data)

        # Fill missing values with the specified constant
        mask = np.isnan(data)
        data[mask] = -1

        # Clip values and apply logarithmic transformation
        data = np.clip(data, np.exp(-6), np.exp(10))
        data = np.log(data)

        # Normalize the data
        data = (data - self.data_mean) / (self.data_std + eps)

        # Convert data to a PyTorch tensor and apply transformations
        data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
        data = config.image_transform(data_tensor)

        # Return the batch data
        return data.numpy()

    def __data_generation(self, index):
        row = self.df.iloc[index]

        X1, X2 = self.__spec_data_gen(row)

        y = np.zeros(6, dtype='float32')
        if self.mode != 'test':
            y = row[label_cols].values.astype(np.float32)

        return X1, X2, y

    def __spec_data_gen(self, row):
        spec = self.all_eegs_512_spec[str(row['eeg_id'])]
        spec = spec[np.newaxis, :, :]
        spec = np.tile(spec, (3, 1, 1))

        new_norm = self.get_datasetwidenorm(row['eeg_id'])
        new_norm = np.tile(new_norm, (3, 1, 1))

        return spec, new_norm


# zfg dataloader

test_dataset = CustomDataset(test_df, config, mode="test")
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False
)
output = test_dataset[0]
X1, X2, y = output["X1"], output["X2"], output["y"]

print(f"X1 shape: {X1.shape}")
print(f"X2 shape: {X2.shape}")
print(f"y shape: {y.shape}")


# zfg model

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次局部注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class SpecNet(nn.Module):

    def __init__(self, backbone_2d, num_classes=6):
        super(SpecNet, self).__init__()

        # 2D efficient
        self.backbone_2d_eeg = timm.create_model(
            config.backbone_2d,
            pretrained=False, pretrained_cfg_overlay=pretrained_cfg_overlay,
            drop_rate=0.1,
            drop_path_rate=0.1

        )
        self.backbone_2d_spec = timm.create_model(
            config.backbone_2d,
            pretrained=False, pretrained_cfg_overlay=pretrained_cfg_overlay,
            drop_rate=0.1,
            drop_path_rate=0.1

        )
        self.features_2d_eeg = nn.Sequential(*list(self.backbone_2d_eeg.children())[:-2])
        self.features_2d_spec = nn.Sequential(*list(self.backbone_2d_spec.children())[:-2])

        # forward
        self.iaaf = iAFF(channels=1280, r=32)
        self.avg_pool_2d = nn.AdaptiveAvgPool2d(1)
        self.flatten_2d = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.backbone_2d_eeg.num_features, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

        self.avg_pool_2d_eeg = nn.AdaptiveAvgPool2d(1)
        self.flatten_2d_eeg = nn.Flatten()
        self.fc1_eeg = nn.Linear(in_features=self.backbone_2d_eeg.num_features, out_features=128)
        self.fc2_eeg = nn.Linear(in_features=128, out_features=num_classes)

        self.avg_pool_2d_spec = nn.AdaptiveAvgPool2d(1)
        self.flatten_2d_spec = nn.Flatten()
        self.fc1_spec = nn.Linear(in_features=self.backbone_2d_eeg.num_features, out_features=128)
        self.fc2_spec = nn.Linear(in_features=128, out_features=num_classes)

    def reshape_input(self, x):
        """
        Reshapes input (128, 256, 8) -> 2*(3, 512, 256) monotone image.
        """
        # === Get spectrograms ===
        spectrograms = [x[:, :, :, i:i + 1] for i in range(4)]
        spectrograms = torch.cat(spectrograms, dim=1)
        spectrograms = torch.cat([spectrograms, spectrograms, spectrograms], dim=3)
        spectrograms = spectrograms.permute(0, 3, 1, 2)

        # === Get EEG spectrograms ===
        eegs = [x[:, :, :, i:i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)
        eegs = torch.cat([eegs, eegs, eegs], dim=3)
        eegs = eegs.permute(0, 3, 1, 2)
        return spectrograms, eegs

    def forward(self, eeg, spec):
        # spec, eeg = self.reshape_input(x)
        spec_emb = self.features_2d_spec(spec)  # [bs, 1280, 16, 16]
        # print(spec_emb.shape)
        eeg_emb = self.features_2d_eeg(eeg)

        res = self.iaaf(spec_emb, eeg_emb)  # [bs, 1280, 16, 16]
        res = self.avg_pool_2d(res)
        res = self.flatten_2d(res)
        res = self.fc1(res)
        res = self.fc2(res)

        spec_emb = self.avg_pool_2d_spec(spec_emb)
        spec_emb = self.flatten_2d_spec(spec_emb)
        spec_emb = self.fc1_spec(spec_emb)
        res_spec = self.fc2_spec(spec_emb)

        eeg_emb = self.avg_pool_2d_eeg(eeg_emb)
        eeg_emb = self.flatten_2d_eeg(eeg_emb)
        eeg_emb = self.fc1_eeg(eeg_emb)
        res_eeg = self.fc2_eeg(eeg_emb)

        return res, res_eeg, res_spec, eeg_emb, spec_emb


model = SpecNet(config.backbone_2d)


# zfg inference function

def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, output in enumerate(tqdm_test_loader):
            X1 = output['X1'].to(device)
            X2 = output['X2'].to(device)
            #             X_spec = X_spec.to(device)
            with torch.no_grad():
                res, res_eeg, res_spec, eeg_emb, spec_emb = model(X1, X2)
                #                 y_preds = res*0.362 + res_eeg*0.326 + res_spec*0.312

                res = softmax(res)
                res_eeg = softmax(res_eeg)
                res_spec = softmax(res_spec)

                y_preds = res * 0.359 + res_eeg * 0.315 + res_spec * 0.326
            preds.append(y_preds.to('cpu').numpy())

    prediction_dict["predictions"] = np.concatenate(preds)
    return prediction_dict


# zfg infer

predictions_v17_s2 = []

for model_weight in model_weights:
    print(model_weight)
    test_dataset = CustomDataset(test_df, config, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False
    )
    #     model = CustomModel(config)
    checkpoint = torch.load(model_weight)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    prediction_dict = inference_function(test_loader, model, device)
    predictions_v17_s2.append(prediction_dict["predictions"])
    torch.cuda.empty_cache()
    gc.collect()

predictions_v17_s2 = np.array(predictions_v17_s2)
predictions_v17_s2 = np.mean(predictions_v17_s2, axis=0)

# zfg v16 s2

model_weights = [f"../hms-harmful-brain-activity-classification/output_1D_2D/wavenet_fold_{x}_s2_best_val.pth" for x in
                 range(5)]


# model_weights.append('/kaggle/input/2d-new/v16_newdata/all_data_epoch_4.pth')

class CustomDataset(Dataset):
    def __init__(
            self, df, config, mode='train', specs=all_spectrograms, eeg_specs=all_eegs_specs, downsample=5,
            augment=config.augment, data_mean=data_mean, data_std=data_std
    ):
        self.df = df
        self.config = config
        self.mode = mode
        self.spectrograms = specs
        self.eeg_spectrograms = eeg_specs
        self.downsample = downsample
        self.augment = augment
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.all_eegs_512_spec = all_eegs_512_spec
        self.data_mean = data_mean
        self.data_std = data_std

    def __len__(self):
        """
        Length of dataset.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Get one item.
        """
        X1, X2, y = self.__data_generation(index)

        # if self.augment and self.mode == 'train':
        #     X_spec = self.__transform(X_spec)

        output = {
            "X1": torch.tensor(X1, dtype=torch.float32),
            "X2": torch.tensor(X2, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32)
        }
        return output

    def get_datasetwidenorm(self, eeg_id):
        """This function will get the batch and preprocess it."""
        # Set a small epsilon to avoid division by zero
        eps = 1e-6

        # Read data from parquet file
        data = all_eegs[eeg_id]
        data = create_spectrogram(data)

        # Fill missing values with the specified constant
        mask = np.isnan(data)
        data[mask] = -1

        # Clip values and apply logarithmic transformation
        data = np.clip(data, np.exp(-6), np.exp(10))
        data = np.log(data)

        # Normalize the data
        data = (data - self.data_mean) / (self.data_std + eps)

        # Convert data to a PyTorch tensor and apply transformations
        data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
        data = config.image_transform(data_tensor)

        # Return the batch data
        return data.numpy()

    def __data_generation(self, index):
        row = self.df.iloc[index]

        X1, X2 = self.__spec_data_gen(row)

        y = np.zeros(6, dtype='float32')
        if self.mode != 'test':
            y = row[label_cols].values.astype(np.float32)

        return X1, X2, y

    def __spec_data_gen(self, row):
        """
        Generates spec data containing batch_size samples.
        """
        X = np.zeros((128, 256, 8), dtype='float32')
        img = np.ones((128, 256), dtype='float32')
        if self.mode == 'test':
            r = 0
        else:
            r = int((row['min'] + row['max']) // 4)
        # r = int(row['spectrogram_label_offset_seconds'] // 2)

        for region in range(4):
            img = self.spectrograms[row.spectrogram_id][r:r + 300, region * 100:(region + 1) * 100].T

            # Log transform spectogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # Standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0
            img = self.eeg_spectrograms[row.eeg_id]
            X[:, :, 4:] = img

        # === Get spectrograms ===
        spectrograms = np.concatenate([X[:, :, i:i + 1] for i in range(4)], axis=0)

        # === Get EEG spectrograms ===
        eegs = np.concatenate([X[:, :, i:i + 1] for i in range(4, 8)], axis=0)

        X = np.concatenate([spectrograms, eegs], axis=1)  # [512, 512, 1]
        X = np.transpose(X, (2, 0, 1))
        X = np.tile(X, (3, 1, 1))

        #         spec = self.all_eegs_512_spec[str(row['eeg_id'])]
        #         spec = spec[np.newaxis, :, :]
        #         spec = np.tile(spec, (3, 1, 1))

        new_norm = self.get_datasetwidenorm(row['eeg_id'])
        new_norm = np.tile(new_norm, (3, 1, 1))

        return X, new_norm

    def __transform(self, img):
        fill_value = tuple(range(img.shape[-1]))
        params1 = {
            "num_masks_x": 1,
            "mask_x_length": (0, 20),  # This line changed from fixed  to a range
            "fill_value": fill_value,
        }
        params2 = {
            "num_masks_y": 1,
            "mask_y_length": (0, 20),
            "fill_value": fill_value,
        }
        params3 = {
            "num_masks_x": (2, 4),
            "num_masks_y": 5,
            "mask_y_length": 8,
            "mask_x_length": (10, 20),
            "fill_value": fill_value,
        }

        transforms = A.Compose([
            A.MixUp(reference_data=self.df.index.to_list(), read_fn=self.read_fn, alpha=1, p=0.5)
        ])
        return transforms(image=img)['image']

    def read_fn(self, index):
        row = self.df.iloc[index]
        img = self.__spec_data_gen(row)

        # img = normalize_image(img)

        global_label = row[label_cols].values.astype(np.float32)

        return {"image": img, "global_label": global_label}


def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, output in enumerate(tqdm_test_loader):
            X1 = output['X1'].to(device)
            X2 = output['X2'].to(device)
            #             X_spec = X_spec.to(device)
            with torch.no_grad():
                res, res_eeg, res_spec, eeg_emb, spec_emb = model(X1, X2)
                #                 y_preds = res*0.362 + res_eeg*0.326 + res_spec*0.312

                res = softmax(res)
                res_eeg = softmax(res_eeg)
                res_spec = softmax(res_spec)

                y_preds = res * 0.356 + res_eeg * 0.31 + res_spec * 0.334
            preds.append(y_preds.to('cpu').numpy())

    prediction_dict["predictions"] = np.concatenate(preds)
    return prediction_dict


predictions_v16_s2 = []

for model_weight in model_weights:
    test_dataset = CustomDataset(test_df, config, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False
    )
    #     model = CustomModel(config)
    checkpoint = torch.load(model_weight)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    prediction_dict = inference_function(test_loader, model, device)
    predictions_v16_s2.append(prediction_dict["predictions"])
    torch.cuda.empty_cache()
    gc.collect()

predictions_v16_s2 = np.array(predictions_v16_s2)
predictions_v16_s2 = np.mean(predictions_v16_s2, axis=0)

# zfg v12 s2

model_weights = [f"../hms-harmful-brain-activity-classification/A_output_1D_2D/wavenet_fold_{x}_s2_best_val.pth" for x
                 in range(5)]


class CustomDataset(Dataset):
    def __init__(
            self, df, config, mode='train', specs=all_spectrograms, eeg_specs=all_eegs_specs, downsample=5,
            augment=config.augment, data_mean=data_mean, data_std=data_std
    ):
        self.df = df
        self.config = config
        self.mode = mode
        self.spectrograms = specs
        self.eeg_spectrograms = eeg_specs
        self.downsample = downsample
        self.augment = augment
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.all_eegs_512_spec = all_eegs_512_spec
        self.data_mean = data_mean
        self.data_std = data_std

    def __len__(self):
        """
        Length of dataset.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Get one item.
        """
        X1, X2, y = self.__data_generation(index)

        # if self.augment and self.mode == 'train':
        #     X_spec = self.__transform(X_spec)

        output = {
            "X1": torch.tensor(X1, dtype=torch.float32),
            "X2": torch.tensor(X2, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32)
        }
        return output

    def get_datasetwidenorm(self, eeg_id):
        """This function will get the batch and preprocess it."""
        # Set a small epsilon to avoid division by zero
        eps = 1e-6

        # Read data from parquet file
        data = all_eegs[eeg_id]
        data = create_spectrogram(data)

        # Fill missing values with the specified constant
        mask = np.isnan(data)
        data[mask] = -1

        # Clip values and apply logarithmic transformation
        data = np.clip(data, np.exp(-6), np.exp(10))
        data = np.log(data)

        # Normalize the data
        data = (data - self.data_mean) / (self.data_std + eps)

        # Convert data to a PyTorch tensor and apply transformations
        data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
        data = config.image_transform(data_tensor)

        # Return the batch data
        return data.numpy()

    def __data_generation(self, index):
        row = self.df.iloc[index]

        X1, X2 = self.__spec_data_gen(row)

        y = np.zeros(6, dtype='float32')
        if self.mode != 'test':
            y = row[label_cols].values.astype(np.float32)

        return X1, X2, y

    def __spec_data_gen(self, row):
        """
        Generates spec data containing batch_size samples.
        """
        X = np.zeros((128, 256, 8), dtype='float32')
        img = np.ones((128, 256), dtype='float32')
        if self.mode == 'test':
            r = 0
        else:
            r = int((row['min'] + row['max']) // 4)
        # r = int(row['spectrogram_label_offset_seconds'] // 2)

        for region in range(4):
            img = self.spectrograms[row.spectrogram_id][r:r + 300, region * 100:(region + 1) * 100].T

            # Log transform spectogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # Standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0
            img = self.eeg_spectrograms[row.eeg_id]
            X[:, :, 4:] = img

        # === Get spectrograms ===
        spectrograms = np.concatenate([X[:, :, i:i + 1] for i in range(4)], axis=0)

        # === Get EEG spectrograms ===
        eegs = np.concatenate([X[:, :, i:i + 1] for i in range(4, 8)], axis=0)

        X = np.concatenate([spectrograms, eegs], axis=1)  # [512, 512, 1]
        X = np.transpose(X, (2, 0, 1))
        X = np.tile(X, (3, 1, 1))

        spec = self.all_eegs_512_spec[str(row['eeg_id'])]
        spec = spec[np.newaxis, :, :]
        spec = np.tile(spec, (3, 1, 1))

        return X, spec

    def __transform(self, img):
        fill_value = tuple(range(img.shape[-1]))
        params1 = {
            "num_masks_x": 1,
            "mask_x_length": (0, 20),  # This line changed from fixed  to a range
            "fill_value": fill_value,
        }
        params2 = {
            "num_masks_y": 1,
            "mask_y_length": (0, 20),
            "fill_value": fill_value,
        }
        params3 = {
            "num_masks_x": (2, 4),
            "num_masks_y": 5,
            "mask_y_length": 8,
            "mask_x_length": (10, 20),
            "fill_value": fill_value,
        }

        transforms = A.Compose([

            A.MixUp(reference_data=self.df.index.to_list(), read_fn=self.read_fn, alpha=1, p=0.5)
        ])
        return transforms(image=img)['image']

    def read_fn(self, index):
        row = self.df.iloc[index]
        img = self.__spec_data_gen(row)

        # img = normalize_image(img)

        global_label = row[label_cols].values.astype(np.float32)

        return {"image": img, "global_label": global_label}


def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, output in enumerate(tqdm_test_loader):
            X1 = output['X1'].to(device)
            X2 = output['X2'].to(device)
            #             X_spec = X_spec.to(device)
            with torch.no_grad():
                res, res_eeg, res_spec, eeg_emb, spec_emb = model(X1, X2)
                #                 y_preds = res*0.362 + res_eeg*0.326 + res_spec*0.312

                res = softmax(res)
                res_eeg = softmax(res_eeg)
                res_spec = softmax(res_spec)

                y_preds = res * 0.35 + res_eeg * 0.323 + res_spec * 0.327
            preds.append(y_preds.to('cpu').numpy())

    prediction_dict["predictions"] = np.concatenate(preds)
    return prediction_dict


predictions_v12_s2 = []

for model_weight in model_weights:
    test_dataset = CustomDataset(test_df, config, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False
    )
    #     model = CustomModel(config)
    checkpoint = torch.load(model_weight)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    prediction_dict = inference_function(test_loader, model, device)
    predictions_v12_s2.append(prediction_dict["predictions"])
    torch.cuda.empty_cache()
    gc.collect()

predictions_v12_s2 = np.array(predictions_v12_s2)
predictions_v12_s2 = np.mean(predictions_v12_s2, axis=0)

# zfg v70 s2

'''
wavenet_fold_0_best_s2_val_by10
'''
#
model_weights = [x for x in glob(
    "../hms-harmful-brain-activity-classification/A_output_1D_2D/wavenet_fold_0_best_s2_val_by10.pth")]


# model_weights

def eeg_from_parquet(parquet_path: str) -> np.ndarray:
    # === Extract middle 50 seconds ===
    eeg = pd.read_parquet(parquet_path, columns=eeg_features)
    rows = len(eeg)
    offset = (rows - 10_000) // 2  # 50 * 200 = 10_000
    eeg = eeg.iloc[offset:offset + 10_000]  # middle 50 seconds, has the same amount of readings to left and right
    # === Convert to numpy ===
    data = np.zeros((10_000, len(eeg_features)))  # create placeholder of same shape with zeros
    for index, feature in enumerate(eeg_features):
        x = eeg[feature].values.astype('float32')  # convert to float32
        mean = np.nanmean(x)  # arithmetic mean along the specified axis, ignoring NaNs
        nan_percentage = np.isnan(x).mean()  # percentage of NaN values in feature
        # === Fill nan values ===
        if nan_percentage < 1:  # if some values are nan, but not all
            x = np.nan_to_num(x, nan=mean)
        else:  # if all values are nan
            x[:] = 0
        data[:, index] = x

    return data


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def sep():
    print("-" * 100)


target_preds = [x + "_pred" for x in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
label_to_num = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
num_to_label = {v: k for k, v in label_to_num.items()}
seed_everything(config.SEED)


class CustomDataset(Dataset):
    def __init__(
            self, df, config, mode='test', eegs=all_eegs, specs=all_spectrograms, eeg_specs=all_eegs_specs,
            downsample=5, data_mean=data_mean, data_std=data_std
    ):
        self.df = df
        self.config = config
        self.mode = mode
        self.eegs = eegs
        self.spectrograms = specs
        self.eeg_spectrograms = eeg_specs
        self.all_eegs_512_spec = all_eegs_512_spec
        self.downsample = downsample
        self.data_mean = data_mean
        self.data_std = data_std

    def __len__(self):
        """
        Length of dataset.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Get one item.
        """
        X_eeg, X_spec, y = self.__data_generation(index)

        # X_eeg = X_eeg[::self.downsample,:]
        output = {
            "X_eeg": torch.tensor(X_eeg, dtype=torch.float32),
            "X_spec": torch.tensor(X_spec, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32)
        }
        return output

    def __data_generation(self, index):
        row = self.df.iloc[index]

        X_eeg = self.__eeg_data_gen(row)
        X_spec = self.__spec_data_gen(row)

        y = np.zeros(6, dtype='float32')
        if self.mode != 'test':
            y = row[label_cols].values.astype(np.float32)

        return X_eeg, X_spec, y

    def resize_func(self, x, target_size=10000, use_percentile_feat=False):
        ch = x.shape[0]
        input_size = x.shape[1]
        pad = target_size - input_size % target_size
        factor = (input_size + pad) / input_size
        x = np.array([ndi.zoom(xi, zoom=factor, mode='reflect') for xi in x])
        x = x.reshape((ch, target_size, -1))
        res = {}
        res['mean'] = np.mean(x, axis=2).reshape(-1, ch)
        res['max'] = np.max(x, axis=2).reshape(-1, ch)
        res['min'] = np.min(x, axis=2).reshape(-1, ch)
        res['med'] = np.median(x, axis=2).reshape(-1, ch)
        res['std'] = np.sqrt(np.var(x, axis=2).reshape(-1, ch))
        if use_percentile_feat:
            for p in [15, 30, 45, 60, 75, 90]:
                res[f"p{p}"] = np.percentile(x, [p], axis=2).reshape(-1, ch)
        return res

    def get_datasetwidenorm(self, eeg_id):
        """This function will get the batch and preprocess it."""
        # Set a small epsilon to avoid division by zero
        eps = 1e-6

        # Read data from parquet file
        data = all_eegs[eeg_id]
        data = create_spectrogram(data)

        # Fill missing values with the specified constant
        mask = np.isnan(data)
        data[mask] = -1

        # Clip values and apply logarithmic transformation
        data = np.clip(data, np.exp(-6), np.exp(10))
        data = np.log(data)

        # Normalize the data
        data = (data - self.data_mean) / (self.data_std + eps)

        # Convert data to a PyTorch tensor and apply transformations
        data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
        data = config.image_transform(data_tensor)

        # Return the batch data
        return data.numpy()

    def __eeg_data_gen(self, row):
        X = np.zeros((10_000, 8), dtype='float32')
        data = self.eegs[row.eeg_id]

        # === Feature engineering ===
        X[:, 0] = data[:, feature_to_index['Fp1']] - data[:, feature_to_index['T3']]
        X[:, 1] = data[:, feature_to_index['T3']] - data[:, feature_to_index['O1']]

        X[:, 2] = data[:, feature_to_index['Fp1']] - data[:, feature_to_index['C3']]
        X[:, 3] = data[:, feature_to_index['C3']] - data[:, feature_to_index['O1']]

        X[:, 4] = data[:, feature_to_index['Fp2']] - data[:, feature_to_index['C4']]
        X[:, 5] = data[:, feature_to_index['C4']] - data[:, feature_to_index['O2']]

        X[:, 6] = data[:, feature_to_index['Fp2']] - data[:, feature_to_index['T4']]
        X[:, 7] = data[:, feature_to_index['T4']] - data[:, feature_to_index['O2']]

        # === Standarize ===
        X = np.clip(X, -1024, 1024)
        X = np.nan_to_num(X, nan=0) / 32.0

        # === Butter Low-pass Filter ===
        X = butter_lowpass_filter(X)
        # X = quantize_data(X, 1)

        return X

    def __spec_data_gen(self, row):
        """
        Generates spec data containing batch_size samples.
        """
        # eeg_id = row['eeg_id']
        # X = self.all_eegs_512_spec[str(eeg_id)]
        X = np.zeros((128, 256, 8), dtype='float32')
        img = np.ones((128, 256), dtype='float32')
        if self.mode == 'test':
            r = 0
        else:
            r = int((row['min'] + row['max']) // 4)
            # r = int(row['spectrogram_label_offset_seconds'] // 2)

        for region in range(4):
            img = self.spectrograms[row.spectrogram_id][r:r + 300, region * 100:(region + 1) * 100].T

            # Log transform spectogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # Standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0
            img = self.eeg_spectrograms[row.eeg_id]
            X[:, :, 4:] = img

        # === Get spectrograms ===
        spectrograms = np.concatenate([X[:, :, i:i + 1] for i in range(4)], axis=0)

        # === Get EEG spectrograms ===
        eegs = np.concatenate([X[:, :, i:i + 1] for i in range(4, 8)], axis=0)

        X = np.concatenate([spectrograms, eegs], axis=1)  # [512, 512, 1]
        X = np.transpose(X, (2, 0, 1))

        spec = self.all_eegs_512_spec[str(row['eeg_id'])]
        spec = spec[np.newaxis, :, :]

        new_norm = self.get_datasetwidenorm(row['eeg_id'])

        X = np.concatenate([X, spec, new_norm], axis=1)
        X = np.tile(X, (3, 1, 1))

        return X


class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=6, r=2):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        # 第二次局部注意力
        self.local_att2 = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class ResNet_1D_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsampling):
        super(ResNet_1D_Block, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.maxpool(out)
        identity = self.downsampling(x)

        out += identity
        return out


class EEGMegaNet(nn.Module):

    def __init__(self, backbone_2d, in_channels_2d, kernels, in_channels=20, fixed_kernel_size=17, num_classes=6):
        super(EEGMegaNet, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True

        # CNN
        self.kernels = kernels
        self.planes = 24
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=(kernel_size),
                                 stride=1, padding=0, bias=False, )
            self.parallel_conv.append(sep_conv)
        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size,
                               stride=2, padding=2, bias=False)
        self.block = self._make_resnet_layer(kernel_size=fixed_kernel_size, stride=1, padding=fixed_kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=4, stride=4, padding=2)
        self.rnn = nn.GRU(input_size=self.in_channels, hidden_size=128, num_layers=1, bidirectional=True)

        self.iaaf = iAFF(channels=3, r=1)

        # 2D efficient
        self.backbone_2d = timm.create_model(
            config.backbone_2d,
            pretrained=False, pretrained_cfg_overlay=pretrained_cfg_overlay,
            drop_rate=0.1,
            drop_path_rate=0.1

        )
        self.features_2d = nn.Sequential(*list(self.backbone_2d.children())[:-2])
        self.avg_pool_2d = nn.AdaptiveAvgPool2d(1)
        self.flatten_2d = nn.Flatten()

        # forward
        self.fc1 = nn.Linear(in_features=self.backbone_2d.num_features, out_features=128)
        self.fc2 = nn.Linear(in_features=736, out_features=128)

        self.fc1d = nn.Linear(in_features=128, out_features=num_classes)
        self.fc2d = nn.Linear(in_features=128, out_features=num_classes)

        self.flatten_final = nn.Flatten()
        self.fc = nn.Linear(in_features=128 * 3, out_features=num_classes)

    def _make_resnet_layer(self, kernel_size, stride, blocks=8, padding=0):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            downsampling = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            )
            layers.append(ResNet_1D_Block(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                          stride=stride, padding=padding, downsampling=downsampling))

        return nn.Sequential(*layers)

    def __reshape_input(self, x):
        """
        Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image.
        """
        # === Get spectrograms ===
        spectrograms = [x[:, :, :, i:i + 1] for i in range(4)]
        spectrograms = torch.cat(spectrograms, dim=1)

        # === Get EEG spectrograms ===
        eegs = [x[:, :, :, i:i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)

        # === Reshape (512,512,3) ===
        if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectrograms, eegs], dim=2)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectrograms

        x = torch.cat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x, spec):

        spec = self.features_2d(spec)
        # print(spec.shape)
        spec = self.avg_pool_2d(spec)
        # print(spec.shape)
        spec = self.flatten_2d(spec)
        # print(spec.shape)

        spec_emb = self.fc1(spec)
        outspec = self.fc2d(spec_emb)
        # print(spec_emb.shape)

        # 1D CNN
        x = x.permute(0, 2, 1)

        out_sep = []

        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)

        out = out.reshape(out.shape[0], -1)
        rnn_out, _ = self.rnn(x.permute(0, 2, 1))
        new_rnn_h = rnn_out[:, -1, :]
        new_out = torch.cat([out, new_rnn_h], dim=1)
        eeg_emb = self.fc2(new_out)
        outseeg = self.fc1d(eeg_emb)

        eeg_feas = eeg_emb.unsqueeze(1).repeat(1, 3, 1)
        spec_feas = spec_emb.unsqueeze(1).repeat(1, 3, 1)
        result = self.iaaf(eeg_feas, spec_feas)

        result = self.flatten_final(result)

        result = self.fc(result)

        return result, eeg_emb, spec_emb, outseeg, outspec


pretrained_cfg_overlay = {
    'file': r"/home/laplace/.cache/huggingface/hub/models--timm--tf_efficientnet_b0.ns_jft_in1k/pytorch_model.bin"}
model = EEGMegaNet(backbone_2d=config.backbone_2d, in_channels_2d=8,
                   kernels=[3, 5, 7, 9],
                   in_channels=8,
                   fixed_kernel_size=5,
                   num_classes=6)


def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, output in enumerate(tqdm_test_loader):
            X_eeg, X_spec, y = output["X_eeg"], output["X_spec"], output["y"]

            X_eeg = X_eeg.to(device)
            X_spec = X_spec.to(device)
            y = y.to(device)

            with torch.no_grad():
                y_preds, eeg_emb, spec_emb, out1d, out2d = model(X_eeg, X_spec)
                #                 print(y_preds)
                y_preds = softmax(y_preds)
                out1d = softmax(out1d)
                out2d = softmax(out2d)
                y_preds = y_preds * 0.371 + out1d * 0.256 + out2d * 0.373
            #             y_preds = softmax(y_preds)
            preds.append(y_preds.to('cpu').numpy())

    prediction_dict["predictions"] = np.concatenate(preds)
    return prediction_dict


predictions_v70_s2 = []

for model_weight in model_weights:
    test_dataset = CustomDataset(test_df, config, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False
    )
    checkpoint = torch.load(model_weight)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    prediction_dict = inference_function(test_loader, model, device)
    predictions_v70_s2.append(prediction_dict["predictions"])
    torch.cuda.empty_cache()
    gc.collect()

predictions_v70_s2 = np.array(predictions_v70_s2)
predictions_v70_s2 = np.mean(predictions_v70_s2, axis=0)

# zfg save submission
submission = pd.read_csv("../hms-harmful-brain-activity-classification/sample_submission.csv")
labels = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
for i in range(len(labels)):
    submission[f'{labels[i]}_vote'] = predictions_v12_s2[:, i] * 0.2271 + predictions_v16_s2[:,
                                                                          i] * 0.2561 + predictions_v17_s2[:,
                                                                                        i] * 0.2633 + predictions_v70_s2[
                                                                                                      :, i] * 0.2535
submission.to_csv("submission.csv", index=None)

submission.iloc[:, -6:].sum(axis=1)

# zfg model8_2

import cupy as cp
import cusignal
from scipy.ndimage import gaussian_filter
import typing as tp
from pathlib import Path
import numpy as np
import torch
from torch import nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

FOLDS = [0, 1, 2]
N_FOLDS = len(FOLDS)


def create_spectrogram_with_cusignal(eeg_data, eeg_id, start, duration=50,
                                     low_cut_freq=0.7, high_cut_freq=20, order_band=5,
                                     spec_size_freq=267, spec_size_time=30,
                                     nperseg_=1500, noverlap_=1483, nfft_=2750,
                                     sigma_gaussian=0.7,
                                     mean_montage_names=4):
    electrode_names = ['LL', 'RL', 'LP', 'RP']

    electrode_pairs = [
        ['Fp1', 'F7', 'T3', 'T5', 'O1'],
        ['Fp2', 'F8', 'T4', 'T6', 'O2'],
        ['Fp1', 'F3', 'C3', 'P3', 'O1'],
        ['Fp2', 'F4', 'C4', 'P4', 'O2']
    ]

    # Filter specifications
    nyquist_freq = 0.5 * 200
    low_cut_freq_normalized = low_cut_freq / nyquist_freq
    high_cut_freq_normalized = high_cut_freq / nyquist_freq

    # Bandpass and notch filter
    bandpass_coefficients = butter(order_band, [low_cut_freq_normalized, high_cut_freq_normalized], btype='band')
    notch_coefficients = iirnotch(w0=60, Q=30, fs=200)

    spec_size = duration * 200
    start = start * 200
    real_start = start + (10_000 // 2) - (spec_size // 2)
    eeg_data = eeg_data.iloc[real_start:real_start + spec_size]

    # Spectrogram parameters
    fs = 200
    nperseg = nperseg_
    noverlap = noverlap_
    nfft = nfft_

    if spec_size_freq <= 0 or spec_size_time <= 0:
        frequencias_size = int((nfft // 2) / 5.15198) + 1
        segmentos = int((spec_size - noverlap) / (nperseg - noverlap))
    else:
        frequencias_size = spec_size_freq
        segmentos = spec_size_time

    spectrogram = cp.zeros((frequencias_size, segmentos, 4), dtype='float32')

    processed_eeg = {}

    for i, name in enumerate(electrode_names):
        cols = electrode_pairs[i]
        processed_eeg[name] = np.zeros(spec_size)
        for j in range(4):
            # Compute differential signals
            signal = cp.array(eeg_data[cols[j]].values - eeg_data[cols[j + 1]].values)

            # Handle NaNs
            mean_signal = cp.nanmean(signal)
            signal = cp.nan_to_num(signal, nan=mean_signal) if cp.isnan(signal).mean() < 1 else cp.zeros_like(signal)

            # Filter bandpass and notch
            signal_filtered = filtfilt(*notch_coefficients, signal.get())
            signal_filtered = filtfilt(*bandpass_coefficients, signal_filtered)
            signal = cp.asarray(signal_filtered)
            # scipy.signal.spectrogram 使用CPU，cusignal使用GPU加速
            frequencies, times, Sxx = cusignal.spectrogram(signal, fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

            # Filter frequency range
            valid_freqs = (frequencies >= 0.59) & (frequencies <= 20)
            frequencies_filtered = frequencies[valid_freqs]
            Sxx_filtered = Sxx[valid_freqs, :]

            # Logarithmic transformation and normalization using Cupy
            spectrogram_slice = cp.clip(Sxx_filtered, cp.exp(-4), cp.exp(6))
            spectrogram_slice = cp.log10(spectrogram_slice)

            normalization_epsilon = 1e-6
            mean = spectrogram_slice.mean(axis=(0, 1), keepdims=True)
            std = spectrogram_slice.std(axis=(0, 1), keepdims=True)
            spectrogram_slice = (spectrogram_slice - mean) / (std + normalization_epsilon)

            spectrogram[:, :, i] += spectrogram_slice
            processed_eeg[f'{cols[j]}_{cols[j + 1]}'] = signal.get()
            processed_eeg[name] += signal.get()

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        if mean_montage_names > 0:
            spectrogram[:, :, i] /= mean_montage_names

    # Convert to NumPy and apply Gaussian filter
    spectrogram_np = cp.asnumpy(spectrogram)
    if sigma_gaussian > 0.0:
        spectrogram_np = gaussian_filter(spectrogram_np, sigma=sigma_gaussian)

    # Filter EKG signal
    ekg_signal_filtered = filtfilt(*notch_coefficients, eeg_data["EKG"].values)
    ekg_signal_filtered = filtfilt(*bandpass_coefficients, ekg_signal_filtered)
    processed_eeg['EKG'] = np.array(ekg_signal_filtered)

    return spectrogram_np, processed_eeg


def create_spectogram_competition(spec_id, seconds_min):
    spec = pd.read_parquet(
        f'../hms-harmful-brain-activity-classification/input/train-raw-csv/test_spectrograms/{spec_id}.parquet')
    inicio = (seconds_min) // 2
    img = spec.fillna(0).values[:, 1:].T.astype("float32")
    img = img[:, inicio:inicio + 300]

    # Log transform and normalize
    img = np.clip(img, np.exp(-4), np.exp(6))
    img = np.log(img)
    eps = 1e-6
    img_mean = img.mean()
    img_std = img.std()
    img = (img - img_mean) / (img_std + eps)

    return img


from scipy.signal import butter, filtfilt, iirnotch

from tqdm import tqdm
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

all_eegs2 = {}
# Make sure the 'images' folder exists
output_folder = 'imagens'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Creation of spectograms on the test base
for i in tqdm(range(len(test)), desc="Processing EEGs"):
    row = test.iloc[i]
    eeg_id = row['eeg_id']
    spec_id = row['spectrogram_id']
    seconds_min = 0
    start_second = 0
    eeg_data = pd.read_parquet(
        f'../hms-harmful-brain-activity-classification/input/train-raw-csv/test_eegs/{eeg_id}.parquet')
    eeg_new_key = eeg_id
    image_50s, _ = create_spectrogram_with_cusignal(eeg_data=eeg_data, eeg_id=eeg_id, start=start_second, duration=50,
                                                    low_cut_freq=0.7, high_cut_freq=20, order_band=5,
                                                    spec_size_freq=267, spec_size_time=501,
                                                    nperseg_=1500, noverlap_=1483, nfft_=2750,
                                                    sigma_gaussian=0.0,
                                                    mean_montage_names=4)
    image_10s, _ = create_spectrogram_with_cusignal(eeg_data=eeg_data, eeg_id=eeg_id, start=start_second, duration=10,
                                                    low_cut_freq=0.7, high_cut_freq=20, order_band=5,
                                                    spec_size_freq=100, spec_size_time=291,
                                                    nperseg_=260, noverlap_=254, nfft_=1030,
                                                    sigma_gaussian=0.0,
                                                    mean_montage_names=4)
    image_10m = create_spectogram_competition(spec_id, seconds_min)

    imagem_final_unico_canal = np.zeros((1068, 501))
    for j in range(4):
        inicio = j * 267
        fim = inicio + 267
        imagem_final_unico_canal[inicio:fim, :] = image_50s[:, :, j]

    imagem_final_unico_canal2 = np.zeros((400, 291))
    for n in range(4):
        inicio = n * 100
        fim = inicio + 100
        imagem_final_unico_canal2[inicio:fim, :] = image_10s[:, :, n]

    imagem_final_unico_canal_resized = cv2.resize(imagem_final_unico_canal, (400, 800), interpolation=cv2.INTER_AREA)
    imagem_final_unico_canal2_resized = cv2.resize(imagem_final_unico_canal2, (300, 400), interpolation=cv2.INTER_AREA)
    eeg_new_resized = cv2.resize(image_10m, (300, 400), interpolation=cv2.INTER_AREA)
    imagem_final = np.zeros((800, 700), dtype=np.float32)
    imagem_final[0:800, 0:400] = imagem_final_unico_canal_resized
    imagem_final[0:400, 400:700] = imagem_final_unico_canal2_resized
    imagem_final[400:800, 400:700] = eeg_new_resized
    imagem_final = imagem_final[::-1]

    imagem_final = cv2.resize(imagem_final, (512, 512), interpolation=cv2.INTER_AREA)

    all_eegs2[eeg_new_key] = imagem_final

    if i == 0:
        plt.figure(figsize=(10, 10))
        plt.imshow(imagem_final, cmap='jet')
        plt.axis('off')
        plt.show()

        print(imagem_final.shape)

# del all_eegs, spectrograms; gc.collect()
test = pd.read_csv(
    '../hms-harmful-brain-activity-classification/input/train-raw-csv/test.csv')
print('Test shape', test.shape)
test.head()


class HMSHBACSpecModel(nn.Module):

    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            in_channels: int,
            num_classes: int,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained,
            num_classes=num_classes, in_chans=in_channels)

    def forward(self, x):
        h = self.model(x)

        return h


FilePath = tp.Union[str, Path]
EegIds = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]


class HMSHBACSpecDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            eeg_ids: tp.Sequence[EegIds],
            image_paths: tp.Sequence[FilePath],
            labels: tp.Sequence[Label],
            transform: A.Compose,
    ):
        self.eeg_ids = eeg_ids
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        eeg_id = self.eeg_ids[index]

        label = self.labels[index]

        data = all_eegs2[eeg_id]  # shape: (128, 256, 4)
        data = data[..., None]

        img = self._apply_transform(data)

        return {"data": img, "target": label}

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img


class CFG:
    model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    img_size = 512
    max_epoch = 9
    batch_size = 32
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience = 5
    seed = 1086
    deterministic = True
    enable_amp = True
    device = "cuda"


def to_device(
        tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
        device: torch.device, *args, **kwargs
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {
            k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


def get_test_path_label(test: pd.DataFrame):
    """Get file path and dummy target info."""

    img_paths = []
    eeg_ids = []
    labels = np.full((len(test), 6), -1, dtype="float32")
    for spec_id in test["spectrogram_id"].values:
        img_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
        img_paths.append(img_path)

        test1 = test[test['spectrogram_id'] == spec_id]
        eeg_id = test1['eeg_id'].values[0]
        eeg_ids.append(eeg_id)
    test_data = {
        "eeg_ids": eeg_ids,
        "image_paths": img_paths,
        "labels": [l for l in labels]}

    return test_data


def get_test_transforms(CFG):
    test_transform = A.Compose([
        A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
        ToTensorV2(p=1.0)
    ])
    return test_transform


def to_device(
        tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
        device: torch.device, *args, **kwargs
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {
            k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


def get_test_path_label(test: pd.DataFrame):
    """Get file path and dummy target info."""

    img_paths = []
    eeg_ids = []
    labels = np.full((len(test), 6), -1, dtype="float32")
    for spec_id in test["spectrogram_id"].values:
        img_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
        img_paths.append(img_path)

        test1 = test[test['spectrogram_id'] == spec_id]
        eeg_id = test1['eeg_id'].values[0]
        eeg_ids.append(eeg_id)
    test_data = {
        "eeg_ids": eeg_ids,
        "image_paths": img_paths,
        "labels": [l for l in labels]}

    return test_data


def get_test_transforms(CFG):
    test_transform = A.Compose([
        A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
        ToTensorV2(p=1.0)
    ])
    return test_transform


test_preds_arr = np.zeros((N_FOLDS, len(test), N_CLASSES))

test_path_label = get_test_path_label(test)
test_transform = get_test_transforms(CFG)
test_dataset = HMSHBACSpecDataset(**test_path_label, transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=CFG.batch_size, num_workers=4, shuffle=False, drop_last=False)

device = torch.device(CFG.device)

for fold_id in range(N_FOLDS):
    print(f"\n[fold {fold_id}]")
    num = 0

    # # get model
    model_path = Path(
        f"../hms-harmful-brain-activity-classification/output_8/stage2/best_model_fold{fold_id}.pth")
    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # # inference
    test_pred = run_inference_loop(model, test_loader, device)
    test_preds_arr[fold_id] = test_pred

    del model
    torch.cuda.empty_cache()
    gc.collect()

test_pred = test_preds_arr.mean(axis=0)

test_pred_df = pd.DataFrame(
    test_pred, columns=CLASSES
)

test_pred_df = pd.concat([test[["eeg_id"]], test_pred_df], axis=1)

smpl_sub = pd.read_csv(DATA / "sample_submission.csv")

sub7 = pd.merge(
    smpl_sub[["eeg_id"]], test_pred_df, on="eeg_id", how="left")

sub_final = pd.DataFrame({"eeg_id": test.eeg_id.values})

labels = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']

# zfg 8个模型的预测结果的融合（8个模型分别预测得到每个类别的得分，八个模型求平均融合，作为最终提交的结果）

for label in labels:
    sub_final[f'{label}_vote'] = (sub2[f'{label}_vote'] + sub1[f'{label}_vote'] + sub4[f'{label}_vote'] + sub5[
        f'{label}_vote'] + sub7[f'{label}_vote']) / 8.0

sub_final.to_csv(f"submission.csv", index=False)
print(f"Submission shape: {sub_final.shape}")

print(sub_final.iloc[:, -6:].sum(axis=1).to_string())
