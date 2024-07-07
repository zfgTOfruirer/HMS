# zfg 第五部分（模型八）：1D+2D
# 2D数据：将其中的spec（128，256，4）与eeg to spec（128，256，4）左右拼接得到（512，512）；
# 将其与zoom spec（512，512）、eeg to spec_2（512，512），转为channel first ，并在第一维度堆叠得到（1，1526，512），第0维度堆叠3次得到2D spec（3，1536，512）。
# 1D数据：
# 1，官方的eeg数据（1000，20）分为4个区域，每个区域相邻电极做差。
# 2，对数据滤波处理得到1D eeg（10000，8）即为50s的8个特征的脑电波数据。
'''
模型八：2D spec（3，1536，512）频谱图数据与1D eeg（10000，8）时序数据，训练1D CNN+GRU和2D EfficientNet_b0融合的多模态模型
'''
import math
import os
import random
import sys
import time
from glob import glob

import cupy as cp
import cusignal
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from scipy.signal import filtfilt, iirnotch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append('../kaggle_kl_div')
from kaggle_kl_div import kaggle_kl_div

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')

import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from scipy import signal

pretrained_cfg_overlay = {
    'file': r"/home/laplace/.cache/huggingface/hub/models--timm--tf_efficientnet_b0.ns_jft_in1k/pytorch_model.bin"}


# zfg Configuration

class config:
    AMP = True
    BATCH_SIZE_TRAIN = 4
    BATCH_SIZE_VALID = 4
    EPOCHS = 10
    FOLDS = 10
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1e7
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    TRAIN_FULL_DATA = False
    VISUALIZE = True
    LR = 0.1
    WEIGHT_DECAY = 0.01
    backbone_2d = 'tf_efficientnet_b0.ns_jft_in1k'
    num_channels = 8
    pretrained = True
    filter_signal = False
    sampling_rate = 200
    lowcut = 0.5  # 0.85  # Нижняя граница в Hz
    highcut = 20  # 25.0  # Верхняя граница в Hz
    filter_order = 2
    image_transform = transforms.Resize((512, 512))
    augment = False
    two_stage = True


class paths:
    OUTPUT_DIR = "../hms-harmful-brain-activity-classification/output_1D_2D/"
    TRAIN_CSV = "../hms-harmful-brain-activity-classification/input_1D_2D/train_small.csv"
    TRAIN_EEGS = "../hms-harmful-brain-activity-classification/input_1D_2D/train_eegs_small/"
    PRE_LOADED_SPECTROGRAMS = '../hms-harmful-brain-activity-classification/output_1D_2D/all_spectrograms.npy'
    PRE_LOADED_EEGS = '../hms-harmful-brain-activity-classification/output_1D_2D/all_eegs_specs.npy'
    TRAIN_SPECTROGRAMS = "../hms-harmful-brain-activity-classification/input_1D_2D/train_spectrograms_small/"


if not os.path.exists(paths.OUTPUT_DIR):
    os.makedirs(paths.OUTPUT_DIR)


# zfg Utils

class AverageMeter(object):

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


def asMinutes(s: float):
    "Convert to minutes."
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since: float, percent: float):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def get_logger(filename=paths.OUTPUT_DIR):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


# zfg 读取官方的eeg
def eeg_from_parquet(parquet_path: str, display: bool = False) -> np.ndarray:
    """
    This function reads a parquet file and extracts the middle 50 seconds of readings. Then it fills NaN values
    with the mean value (ignoring NaNs).
    :param parquet_path: path to parquet file.
    :param display: whether to display EEG plots or not.
    :return data: np.array of shape  (time_steps, eeg_features) -> (10_000, 8)
    """
    # === Extract middle 50 seconds ===
    eeg = pd.read_parquet(parquet_path, columns=eeg_features)  # zfg 从文件中得到eeg
    rows = len(eeg)
    offset = (rows - 10_000) // 2  # zfg 计算提取50s数据的偏移量 50 * 200 = 10_000
    eeg = eeg.iloc[
          offset:offset + 10_000]  # zfg 提取50s的数据 # middle 50 seconds, has the same amount of readings to left and right
    if display:
        plt.figure(figsize=(10, 5))
        offset = 0
    # === Convert to numpy ===
    data = np.zeros((10_000, len(eeg_features)))  # zfg 创建一个10_000 * 20的数组  50秒的20个电极特征
    for index, feature in enumerate(eeg_features):  # zfg 遍历20个电极特征,数据预处理,并绘制20条脑电波
        x = eeg[feature].values.astype('float32')  # convert to float32
        mean = np.nanmean(x)  # arithmetic mean along the specified axis, ignoring NaNs
        nan_percentage = np.isnan(x).mean()  # percentage of NaN values in feature
        # === Fill nan values ===
        if nan_percentage < 1:  # if some values are nan, but not all
            x = np.nan_to_num(x, nan=mean)
        else:  # if all values are nan
            x[:] = 0
        data[:, index] = x
        if display:
            if index != 0:
                offset += x.max()
            plt.plot(range(10_000), x - offset, label=feature)
            offset -= x.min()
    if display:
        plt.legend()
        name = parquet_path.split('/')[-1].split('.')[0]
        plt.yticks([])
        plt.title(f'EEG {name}', size=16)
        plt.show()
    return data


def plot_spectrogram(spectrogram_path: str):
    """
    Source: https://www.kaggle.com/code/mvvppp/hms-eda-and-domain-journey
    Visualize spectogram recordings from a parquet file.
    :param spectrogram_path: path to the spectogram parquet.
    """
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


# A40 80G 48G 120G 16CPU
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
    spec = pd.read_parquet(paths.TRAIN_SPECTROGRAMS + f'{spec_id}.parquet')
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
    spectogram_0 = np.concatenate((LL, LP, RP, RL), axis=0)
    spectogram = cv2.resize(spectogram_0, (512, 512), interpolation=cv2.INTER_AREA)
    return spectogram


# zfg 数据的标签处理
target_preds = [x + "_pred" for x in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
label_to_num = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
num_to_label = {v: k for k, v in label_to_num.items()}
LOGGER = get_logger()
seed_everything(config.SEED)  # zfg 设置一个全局的随机种子 确保实验的可重复性

# zfg load


# zfg read one eeg parquet

eeg_df = pd.read_parquet(paths.TRAIN_EEGS + "261945270.parquet")  # zfg 读取存储在 Parquet 文件中的数据
eeg_features = eeg_df.columns
print(f'There are {len(eeg_features)} raw eeg features')
print(list(eeg_features))
feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}

# zfg data pre-processing

df = pd.read_csv('../hms-harmful-brain-activity-classification/A_input_1D_2D/train.csv')

label_cols = df.columns[-6:]  # 最后6列

TARGETS = df.columns[-6:]
print('Train shape:', df.shape)
print('Targets', list(TARGETS))
df.head()

train_df = df.drop_duplicates(
    subset=['eeg_id', 'seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote'])
print('Train shape', train_df.shape)

train_df['total_evaluators'] = train_df[
    ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].sum(axis=1)
y_data = train_df[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)  # zfg 标签投票得到的概率数值
train_df[TARGETS] = y_data  # zfg TARGETS 这一列的数据 设置为概率
train_df['target'] = train_df['expert_consensus']
train_df = train_df.reset_index()  # zfg 索引重新设置0开始
train_df.head()
# zfg 在train_df 中挑选出数据
# 定义文件路径和目标列名
file_train = "../hms-harmful-brain-activity-classification/A_input_1D_2D/train_small.csv"
target_column = "label_id"
# zfg 读取train_small.csv文件并获取label_id列的所有数据
label_ids = pd.read_csv(file_train)[target_column].tolist()
print(f"Read {len(label_ids)} label_ids from {file_train}")

# zfg 使用label_ids列表进行索引匹配
train_sample = train_df[train_df[target_column].isin(label_ids)].copy()  # zfg label_id 图像文件中的命名与表格匹配
print(f"Number of rows in train_sample: {train_sample.shape[0]}")
train_df = train_sample.copy()
eeg_ids = train_df.eeg_id.unique()
# zfg

# zfg   read all eeg parquet

CREATE_EEGS = True
all_eegs = {}
visualize = 1
eeg_paths = glob(paths.TRAIN_EEGS + "*.parquet")
eeg_ids = train_df.eeg_id.unique()

for i, eeg_id in tqdm(enumerate(eeg_ids)):
    # Save EEG to Python dictionary of numpy arrays
    eeg_path = paths.TRAIN_EEGS + str(eeg_id) + ".parquet"
    data = eeg_from_parquet(eeg_path, display=i < visualize)  # zfg 读取官方的eeg
    all_eegs[eeg_id] = data

    if i == visualize:
        if CREATE_EEGS:
            print(f'Processing {train_df.eeg_id.nunique()} eeg parquets... ', end='')
        else:
            print(f'Reading {len(eeg_ids)} eeg NumPys from disk.')
            break

# 遍历字典并打印每个键和值的尺寸
for key, value in all_eegs.items():
    print(f'Key: {key}, Value shape: {value.shape}')
# zfg read train spectrograms

READ_EEG_SPEC_FILES = False

paths_eegs = glob(paths.TRAIN_EEGS + "*.npy")
print(f'There are {len(paths_eegs)} EEG spectrograms')

if READ_EEG_SPEC_FILES:
    all_eegs = {}
    for file_path in tqdm(paths_eegs):
        eeg_id = file_path.split("/")[-1].split(".")[0]
        eeg_spectogram = np.load(file_path)
        all_eegs[eeg_id] = eeg_spectogram
else:
    all_eegs_specs = np.load(paths.PRE_LOADED_EEGS, allow_pickle=True).item()

print(len(all_eegs_specs))

READ_SPEC_FILES = False

paths_spectrograms = glob(paths.TRAIN_SPECTROGRAMS + "*.parquet")
print(f'There are {len(paths_spectrograms)} spectrogram parquets')

if READ_SPEC_FILES:
    all_spectrograms = {}
    for file_path in tqdm(paths_spectrograms):
        aux = pd.read_parquet(file_path)
        name = int(file_path.split("/")[-1].split('.')[0])
        all_spectrograms[name] = aux.iloc[:, 1:].values
        del aux
else:
    all_spectrograms = np.load(paths.PRE_LOADED_SPECTROGRAMS, allow_pickle=True).item()

if config.VISUALIZE:
    idx = np.random.randint(0, len(paths_spectrograms))
    spectrogram_path = paths_spectrograms[idx]
    plot_spectrogram(spectrogram_path)

data_mean = 0.00011579196009429602
data_std = 4.5827806440634316

# zfg validation

from sklearn.model_selection import GroupKFold

train_df = train_df.reset_index(drop=True)
gkf = GroupKFold(n_splits=config.FOLDS)  # zfg 多折交叉验证
for fold, (train_index, valid_index) in enumerate(gkf.split(train_df, train_df.target, train_df.patient_id)):
    train_df.loc[valid_index, "fold"] = int(fold)

# zfg butter loww-pass filter

from scipy.signal import butter

frequencies = [1, 2, 4, 8, 16][::-1]  # frequencies in Hz
x = [all_eegs[eeg_ids[0]][:, 0]]  # select one EEG feature
b = np.array(x)
print(b.shape)

plt.figure(figsize=(12, 8))
plt.plot(range(10_000), x[0], label='without filter')
for k in range(1, len(x)):
    plt.plot(range(10_000), x[k] - k * (x[0].max() - x[0].min()), label=f'with filter {frequencies[k - 1]}Hz')

plt.legend()
plt.yticks([])
plt.title('Butter Low-Pass Filter Examples', size=18)
plt.show()

ALL_EEGS_512_SPEC_CREATE = True
# 重置索引为从 0 开始的连续索引

if ALL_EEGS_512_SPEC_CREATE:
    all_eegs_512_spec = {}
    # for i, eeg_id in tqdm(enumerate(eeg_ids)):
    for index in tqdm(train_df.index):
        row = train_df.iloc[index]
        eeg_id = row['eeg_id']
        spec_id = row['spectrogram_id']
        # eeg_data = all_eegs[row['eeg_id']]
        eeg_data = pd.read_parquet(
            f'../hms-harmful-brain-activity-classification/A_input_1D_2D/train_eegs_small/{eeg_id}.parquet')
        all_eegs_512_spec[str(eeg_id)] = create_512_spec(eeg_data, eeg_id, spec_id)
        np.savez_compressed(
            '../hms-harmful-brain-activity-classification/A_input_1D_2D/train_eegs_small/all_eegs_512_spec_unique.npz',
            **all_eegs_512_spec)
else:
    all_eegs_512_spec = np.load('all_eegs_512_spec_unique.npz')

plt.figure(figsize=(10, 10))
plt.imshow(all_eegs_512_spec['481358497'], cmap='jet')
plt.axis('off')
plt.show()


# zfg dataset
class CustomDataset(Dataset):
    def __init__(
            self, df, config, mode='train', eegs=all_eegs, specs=all_spectrograms, eeg_specs=all_eegs_specs,
            downsample=5, augment=config.augment, data_mean=data_mean, data_std=data_std
    ):
        self.df = df
        self.config = config
        self.batch_size = self.config.BATCH_SIZE_TRAIN
        self.mode = mode
        self.eegs = eegs
        self.spectrograms = specs
        self.eeg_spectrograms = eeg_specs
        self.all_eegs_512_spec = all_eegs_512_spec
        self.downsample = downsample
        self.augment = augment
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
        print(data)
        if data.shape[0] < 10000:
            raise ValueError(f"Data length for eeg_id {row['eeg_id']} is less than 10000")
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
        # X = butter_lowpass_filter(X)
        return X

    def __spec_data_gen(self, row):
        """
        Generates spec data containing batch_size samples.
        """
        X = np.zeros((128, 256, 8), dtype='float32')
        img = np.ones((128, 256), dtype='float32')
        if self.mode == 'test':
            r = 0
        else:
            r = int(row['spectrogram_label_offset_seconds'] // 2)
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
            print(self.eeg_spectrograms.keys())
            img = self.eeg_spectrograms[row.eeg_id]
            X[:, :, 4:] = img
        # === Get spectrograms ===
        spectrograms = np.concatenate([X[:, :, i:i + 1] for i in range(4)], axis=0)  # [512, 256, 1]
        # === Get EEG spectrograms ===
        eegs = np.concatenate([X[:, :, i:i + 1] for i in range(4, 8)], axis=0)  # [512, 256, 1]
        # eegs = np.transpose(eegs, (2, 0, 1))
        X = np.concatenate([spectrograms, eegs], axis=1)  # [512, 512, 1]
        X = np.transpose(X, (2, 0, 1))  # [1, 512, 512]
        spec = self.all_eegs_512_spec[str(row['eeg_id'])]  # [512, 512]
        spec = spec[np.newaxis, :, :]  # [1, 512, 512]
        new_norm = self.get_datasetwidenorm(row['eeg_id'])  # [1, 512, 512]
        # print(new_norm.shape)
        # v69
        X = np.concatenate([X, spec, new_norm], axis=1)  # #[1, 1536, 512]
        X = np.tile(X, (3, 1, 1))  # [3, 1536, 512]
        return X


# zfg dataloader

train_dataset = CustomDataset(train_df, config, mode="train")
train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE_TRAIN,
    shuffle=False,
    num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
)
output = train_dataset[0]
X_eeg, X_spec, y = output["X_eeg"], output["X_spec"], output["y"]  # zfg 多模态数据准备
print(f"X_eeg shape: {X_eeg.shape}")
print(f"X_spec shape: {X_spec.shape}")
print(f"y shape: {y.shape}")


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
        # print(xa.shape)  [2, 3, 128]
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        # print(xl.shape)  [2, 3, 128]
        # print(xg.shape)  [2, 3, 1]
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


import timm


class EEGMegaNet(nn.Module):
    """
    EEGMegaNet类结合了1D卷积神经网络（CNN）和2D卷积神经网络（CNN），以及循环神经网络（RNN），
    旨在处理 EEG 数据和spectrogram数据。它包含多个卷积层、批量归一化层、激活函数层、池化层、全连接层
    以及门控循环单元（GRU）。这个网络设计用于在EEG数据上执行分类任务。
    """

    def __init__(self, backbone_2d, kernels, in_channels=20, fixed_kernel_size=17, num_classes=6):
        super(EEGMegaNet, self).__init__()
        # 初始化一些标志变量
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True

        # 初始化1D CNN的相关参数和层
        # 初始化卷积核的数量
        self.kernels = kernels
        # 定义输出平面的数量为24
        self.planes = 24
        # 初始化并行卷积模块列表
        self.parallel_conv = nn.ModuleList()
        # 初始化输入通道的数量
        self.in_channels = in_channels
        # 根据kernels列表中的大小，初始化一系列并行的1D卷积层
        for i, kernel_size in enumerate(list(self.kernels)):
            self.parallel_conv.append(
                nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=(kernel_size),
                          stride=1, padding=0, bias=False))
        # 使用批量归一化和ReLU激活函数增强1D CNN的性能
        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        # 添加一个固定的1D卷积层和残差块
        self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size,
                               stride=2, padding=2, bias=False)
        self.block = self._make_resnet_layer(kernel_size=fixed_kernel_size, stride=1, padding=fixed_kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        # 应用平均池化和循环神经网络用于时序数据的处理
        self.avgpool = nn.AvgPool1d(kernel_size=4, stride=4, padding=2)
        self.rnn = nn.GRU(input_size=self.in_channels, hidden_size=128, num_layers=1, bidirectional=True)

        # 引入iAFF层用于特征融合
        self.iaaf = iAFF(channels=3, r=1)

        # 初始化2D CNN，用于处理来自不同视角的EEG数据
        # 初始化2D CNN的相关部分
        self.backbone_2d = timm.create_model(
            backbone_2d,
            pretrained=False,
            drop_rate=0.1,
            drop_path_rate=0.1
        )
        # 保留2D CNN的前几层作为特征提取器
        self.features_2d = nn.Sequential(*list(self.backbone_2d.children())[:-2])
        self.avg_pool_2d = nn.AdaptiveAvgPool2d(1)
        self.flatten_2d = nn.Flatten()

        # 设置全连接层用于最终的分类决策
        # 初始化全连接层
        self.fc1 = nn.Linear(in_features=self.backbone_2d.num_features, out_features=128)
        self.fc2 = nn.Linear(in_features=736, out_features=128)

        self.fc1d = nn.Linear(in_features=128, out_features=num_classes)
        self.fc2d = nn.Linear(in_features=128, out_features=num_classes)

        self.flatten_final = nn.Flatten()
        # 将1D和2D CNN的输出合并，然后通过全连接层进行分类
        self.fc = nn.Linear(in_features=128 * 3, out_features=num_classes)

    def _make_resnet_layer(self, kernel_size, stride, blocks=8, padding=0):
        layers = []  # 存储ResNet块的列表
        downsample = None  # 默认下采样操作为None
        base_width = self.planes  # 基础宽度，即输入通道数
        # 根据指定的blocks数量创建ResNet块
        for i in range(blocks):
            # 当前ResNet块的下采样操作，仅在第一个块（i==0）时设置
            if i == 0:
                downsample = nn.Sequential(
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                )
            layers.append(ResNet_1D_Block(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                          stride=stride, padding=padding, downsampling=downsample))
        return nn.Sequential(*layers)  # 返回包含所有ResNet块的序列模型

    def forward(self, x, spec):
        # 处理2D spectrogram数据
        # 处理2D spectrogram数据
        spec = self.features_2d(spec)
        spec = self.avg_pool_2d(spec)
        spec = self.flatten_2d(spec)
        spec_emb = self.fc1(spec)
        outspec = self.fc2d(spec_emb)
        # 处理1D EEG数据
        # 转换维度，将通道放在最后，以适应后续的卷积操作
        x = x.permute(0, 2, 1)
        # 通过多个并行卷积层处理EEG数据
        out_sep = [self.parallel_conv[i](x) for i in range(len(self.kernels))]
        # 合并所有卷积层的输出
        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        # 应用ResNet层
        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)
        # 展平输出形状，准备输入RNN
        out = out.reshape(out.shape[0], -1)

        # 通过RNN处理序列数据，获取最后一时刻的输出作为特征
        rnn_out, _ = self.rnn(x.permute(0, 2, 1))
        new_rnn_h = rnn_out[:, -1, :]
        # 合并EEG卷积特征和RNN特征
        new_out = torch.cat([out, new_rnn_h], dim=1)
        eeg_emb = self.fc2(new_out)
        outseeg = self.fc1d(eeg_emb)
        # 扩展特征维度，以进行交叉注意力操作
        eeg_feas = eeg_emb.unsqueeze(1).repeat(1, 3, 1)
        spec_feas = spec_emb.unsqueeze(1).repeat(1, 3, 1)
        # 通过交叉注意力机制融合EEG和spectrogram特征
        result = self.iaaf(eeg_feas, spec_feas)
        result = self.flatten_final(result)
        result = self.fc(result)

        return result, eeg_emb, spec_emb, outseeg, outspec


pretrained_cfg_overlay = {
    'file': r"/home/laplace/.cache/huggingface/hub/models--timm--tf_efficientnet_b0.ns_jft_in1k/pytorch_model.bin"}
model = EEGMegaNet(backbone_2d=config.backbone_2d,
                   kernels=[3, 5, 7, 9],
                   in_channels=8,
                   fixed_kernel_size=5,
                   num_classes=6)

import gc

iot = torch.randn(2, 10000, 8)
# iot2 = torch.randn(2, 128, 256, 8)
iot2 = torch.randn(2, 3, 1536, 512)
result, eeg_emb, spec_emb, outseeg, outspec = model(iot, iot2)
print(result.shape)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# zfg scheduler


from torch.optim.lr_scheduler import OneCycleLR

EPOCHS = 5
BATCHES = len(train_loader)
steps = []
lrs = []
optim_lrs = []
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy="cos",
    final_div_factor=100,
)
for epoch in range(EPOCHS):
    epoch_lrs = []
    for batch in range(BATCHES):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
        steps.append(epoch * BATCHES + batch)

max_lr = max(lrs)
min_lr = min(lrs)
print(f"Maximum LR: {max_lr} | Minimum LR: {min_lr}")
plt.figure()
plt.plot(steps, lrs, label='OneCycle')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.show()

# zfg loss function


import torch.nn.functional as F

# === Reduction = "mean" ===
criterion = nn.KLDivLoss(reduction="mean")
y_pred = F.log_softmax(torch.randn(6, 2, requires_grad=True), dim=1)
y_true = F.softmax(torch.rand(6, 2), dim=1)
print(f"Predictions: {y_pred}")
print(f"Targets: {y_true}")
output = criterion(y_pred, y_true)
print(f"Output: {output}")

print("\n", "=" * 100, "\n")

# === Reduction = "batchmean" ===
criterion = nn.KLDivLoss(reduction="batchmean")  # zfg 每个批次的平均损失
y_pred = F.log_softmax(torch.randn(2, 6, requires_grad=True), dim=1)
y_true = F.softmax(torch.rand(2, 6), dim=1)
print(f"Predictions: {y_pred}")
print(f"Targets: {y_true}")
output = criterion(y_pred, y_true)
print(f"Output: {output}")


# zfg train and validation functions

def train_epoch(train_loader, model, optimizer, epoch, scheduler, device):
    """One epoch training pass."""
    model.train()
    criterion = nn.KLDivLoss(reduction="batchmean")
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)
    losses = AverageMeter()
    criterion_contrastive_loss = nn.CosineEmbeddingLoss()
    start = end = time.time()
    global_step = 0

    # ========== ITERATE OVER TRAIN BATCHES ============
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, batch in enumerate(tqdm_train_loader):
            X_eeg = batch.pop("X_eeg").to(device)  # send inputs to `device`
            X_spec = batch.pop("X_spec").to(device)  # send inputs to `device`
            y = batch.pop("y").to(device)  # send labels to `device`
            batch_size = y.size(0)
            with torch.cuda.amp.autocast(enabled=config.AMP):
                y_preds, eeg_emb, spec_emb, out1d, out2d = model(X_eeg, X_spec)
                loss = criterion(F.log_softmax(y_preds, dim=1), y)
                loss2 = criterion(F.log_softmax(out1d, dim=1), y)
                loss3 = criterion(F.log_softmax(out2d, dim=1), y)

                embedding_1d = torch.nn.functional.normalize(eeg_emb, p=2, dim=1)
                embedding_2d = torch.nn.functional.normalize(spec_emb, p=2, dim=1)
                contrastive_target = torch.ones(embedding_1d.size(0)).to(device)  # Assuming all pairs are similar
                contrastive_loss1 = criterion_contrastive_loss(embedding_1d, embedding_2d, contrastive_target)

                loss = loss + loss2 + loss3 + contrastive_loss1

            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                # if epoch != 'last':
                scheduler.step()
            end = time.time()

    return losses.avg


def valid_epoch(valid_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    losses = AverageMeter()
    criterion = nn.KLDivLoss(reduction="batchmean")
    criterion_contrastive_loss = nn.CosineEmbeddingLoss()
    prediction_dict = {}
    preds, out1d_list, out2d_list = [], [], []
    start = end = time.time()
    with tqdm(valid_loader, unit="valid_batch", desc='Validation') as tqdm_valid_loader:
        for step, batch in enumerate(tqdm_valid_loader):
            X_eeg = batch.pop("X_eeg").to(device)  # send inputs to `device`
            X_spec = batch.pop("X_spec").to(device)  # send inputs to `device
            y = batch.pop("y").to(device)
            batch_size = y.size(0)
            with torch.no_grad():
                y_preds, eeg_emb, spec_emb, out1d, out2d = model(X_eeg, X_spec)
                loss = criterion(F.log_softmax(y_preds, dim=1), y)
                loss2 = criterion(F.log_softmax(out1d, dim=1), y)
                loss3 = criterion(F.log_softmax(out2d, dim=1), y)
                embedding_1d = torch.nn.functional.normalize(eeg_emb, p=2, dim=1)
                embedding_2d = torch.nn.functional.normalize(spec_emb, p=2, dim=1)
                contrastive_target = torch.ones(embedding_1d.size(0)).to(device)  # Assuming all pairs are similar
                contrastive_loss1 = criterion_contrastive_loss(embedding_1d, embedding_2d, contrastive_target)
                loss = loss + loss2 + loss3 + contrastive_loss1
            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            y_preds = softmax(y_preds)
            out1d = softmax(out1d)
            out2d = softmax(out2d)
            preds.append(y_preds.to('cpu').numpy())
            out1d_list.append(out1d.to('cpu').numpy())
            out2d_list.append(out2d.to('cpu').numpy())
            end = time.time()
    prediction_dict["predictions"] = np.concatenate(preds)
    prediction_dict["out1d"] = np.concatenate(out1d_list)
    prediction_dict["out2d"] = np.concatenate(out2d_list)
    return losses.avg, prediction_dict


# zfg train loop

def train_loop(df, fold, two_stage=False):
    LOGGER.info(f"========== Fold: {fold} training ==========")

    # ======== SPLIT ==========
    if two_stage:
        train_folds = df[(df['fold'] != fold) & (df['total_evaluators'] >= 10)].reset_index(drop=True)
    else:
        train_folds = df[(df['fold'] != fold)].reset_index(drop=True)

    valid_folds = df[(df['fold'] == fold)].reset_index(drop=True)

    # ======== DATASETS ==========
    train_dataset = CustomDataset(train_folds, config, mode="train")
    valid_dataset = CustomDataset(valid_folds, config, mode="valid")

    # ======== DATALOADERS ==========
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.BATCH_SIZE_VALID,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)

    # ======== MODEL ==========
    model = EEGMegaNet(backbone_2d=config.backbone_2d, kernels=[3, 5, 7, 9], in_channels=8, fixed_kernel_size=5,
                       num_classes=6)

    if two_stage:
        checkpoint = torch.load(f"{paths.OUTPUT_DIR}/wavenet_fold_{fold}_best.pth")
        model.load_state_dict(checkpoint["model"])

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )

    # ======= LOSS ==========
    criterion = nn.KLDivLoss(reduction="batchmean")

    best_loss = np.inf
    # ====== ITERATE EPOCHS ========
    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # ======= TRAIN ==========
        avg_train_loss = train_epoch(train_loader, model, optimizer, epoch, scheduler, device)

        # ======= EVALUATION ==========
        avg_val_loss, prediction_dict = valid_epoch(valid_loader, model, device)
        predictions = prediction_dict["predictions"]
        out1d = prediction_dict["out1d"]
        out2d = prediction_dict["out2d"]

        # ======= SCORING ==========
        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  lr: {scheduler.get_last_lr()[0]:.4f}  time: {elapsed:.0f}s')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch} - Save Best Loss: {best_loss:.4f} Model')
            if two_stage:
                torch.save({'model': model.state_dict(),
                            'predictions': predictions,
                            'out1d': out1d,
                            'out2d': out2d
                            },
                           f"{paths.OUTPUT_DIR}/wavenet_fold_{fold}_best_s2_val_by10.pth")
            else:
                torch.save({'model': model.state_dict(),
                            'predictions': predictions,
                            'out1d': out1d,
                            'out2d': out2d
                            },
                           f"{paths.OUTPUT_DIR}/wavenet_fold_{fold}_best.pth")
    if two_stage:
        loaded_model = torch.load(f"{paths.OUTPUT_DIR}/wavenet_fold_{fold}_best_s2_val_by10.pth",
                                  map_location=torch.device('cpu'))
    else:
        loaded_model = torch.load(f"{paths.OUTPUT_DIR}/wavenet_fold_{fold}_best.pth",
                                  map_location=torch.device('cpu'))

    predictions = loaded_model['predictions']
    out1d = loaded_model['out1d']
    out2d = loaded_model['out2d']
    valid_folds[[i + 'preds' for i in target_preds]] = predictions
    valid_folds[[i + 'out1d' for i in target_preds]] = out1d
    valid_folds[[i + 'out2d' for i in target_preds]] = out2d

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds


# zfg train

def get_result(oof_df):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    labels = torch.tensor(oof_df[label_cols].values)
    preds = torch.tensor(oof_df[target_preds].values)
    preds = F.log_softmax(preds, dim=1)
    result = kl_loss(preds, labels)
    return result


oof_df = pd.DataFrame()
for fold in range(config.FOLDS):
    if fold in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        _oof_df = train_loop(train_df, fold)
        oof_df = pd.concat([oof_df, _oof_df])
        LOGGER.info(f"========== Fold {fold} finished =======sep===")
oof_df = oof_df.reset_index(drop=True)
oof_df.to_csv(f'{paths.OUTPUT_DIR}/oof_df.csv', index=False)

pred1 = oof_df[[i + 'preds' for i in target_preds]]
pred2 = oof_df[[i + 'out1d' for i in target_preds]]
pred3 = oof_df[[i + 'out2d' for i in target_preds]]
p1 = pd.DataFrame(pred1.copy())
p2 = pd.DataFrame(pred2.copy())
p3 = pd.DataFrame(pred3.copy())
p1.columns = label_cols
p2.columns = label_cols
p3.columns = label_cols

# === Pre-process OOF ===
label_cols = label_cols.tolist()
gt = train_df[["eeg_id"] + label_cols]
gt.sort_values(by="eeg_id", inplace=True)
gt.reset_index(inplace=True, drop=True)


def cal_score(i):
    i = i.iloc[oof_df.index]
    i['eeg_id'] = oof_df['eeg_id']
    preds = i[["eeg_id"] + label_cols]
    preds.columns = ["eeg_id"] + label_cols
    preds.sort_values(by="eeg_id", inplace=True)
    preds.reset_index(inplace=True, drop=True)

    y_trues = gt[label_cols]
    y_preds = preds[label_cols]

    oof = pd.DataFrame(y_preds.copy())
    oof['id'] = np.arange(len(oof))

    true = pd.DataFrame(y_trues.copy())
    true['id'] = np.arange(len(true))

    cv = kaggle_kl_div.score(solution=true, submission=oof, row_id_column_name='id')

    return cv


cv_list = []
res = []
for i in [p1, p2, p3]:
    # for i in [p1]:
    cv = cal_score(i)
    res.append(round(cv, 3))
    cv_list.append(1 / cv)

optimized_weights = [round(i / sum(cv_list), 3) for i in cv_list]

w1 = optimized_weights[0]
w2 = optimized_weights[1]
w3 = optimized_weights[2]
fused_pred = (w1 * p1 + w2 * p2 + w3 * p3) / (w1 + w2 + w3)
fused_pred['eeg_id'] = oof_df['eeg_id']
print(optimized_weights)
print(round(cal_score(fused_pred), 3))
print(res)

config.two_stage = True
if config.two_stage:
    oof_df = pd.DataFrame()
    for fold in range(config.FOLDS):
        if fold in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            _oof_df = train_loop(train_df, fold, config.two_stage)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== Fold {fold} finished =======sep===")
    oof_df = oof_df.reset_index(drop=True)
    oof_df.to_csv(f'{paths.OUTPUT_DIR}/oof_df_s2_val.csv', index=False)

if config.two_stage:

    pred1 = oof_df[[i + 'preds' for i in target_preds]]
    pred2 = oof_df[[i + 'out1d' for i in target_preds]]
    pred3 = oof_df[[i + 'out2d' for i in target_preds]]
    p1 = pd.DataFrame(pred1.copy())
    p2 = pd.DataFrame(pred2.copy())
    p3 = pd.DataFrame(pred3.copy())
    p1.columns = label_cols
    p2.columns = label_cols
    p3.columns = label_cols

    import sys

    sys.path.append('kaggle-kl-div')
    from kaggle_kl_div import score

    # === Pre-process OOF ===

    gt = train_df[["eeg_id"] + label_cols]

    gt.sort_values(by="eeg_id", inplace=True)
    gt.reset_index(inplace=True, drop=True)


    def cal_score(i):
        i = i.iloc[oof_df.index]
        i['eeg_id'] = oof_df['eeg_id']
        preds = i[["eeg_id"] + label_cols]
        preds.columns = ["eeg_id"] + label_cols
        preds.sort_values(by="eeg_id", inplace=True)
        preds.reset_index(inplace=True, drop=True)

        y_trues = gt[label_cols]
        y_preds = preds[label_cols]

        oof = pd.DataFrame(y_preds.copy())
        oof['id'] = np.arange(len(oof))

        true = pd.DataFrame(y_trues.copy())
        true['id'] = np.arange(len(true))

        cv = score(solution=true, submission=oof, row_id_column_name='id')

        return cv


    cv_list = []
    res = []
    for i in [p1, p2, p3]:
        # for i in [p1]:
        cv = cal_score(i)
        res.append(round(cv, 3))
        cv_list.append(1 / cv)

    optimized_weights = [round(i / sum(cv_list), 3) for i in cv_list]

    w1 = optimized_weights[0]
    w2 = optimized_weights[1]
    w3 = optimized_weights[2]
    fused_pred = (w1 * p1 + w2 * p2 + w3 * p3) / (w1 + w2 + w3)
    fused_pred['eeg_id'] = oof_df['eeg_id']
    print(optimized_weights)
    print(round(cal_score(fused_pred), 3))
    print(res)
