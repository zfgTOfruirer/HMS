# zfg 模型七的数据：
# 采用原始的raw eeg 20个电极的相邻电极脑电波数据做差，通过巴特沃斯带通滤波器低通滤波处理得到（10000，18）即为50s的18个特征的脑电波数据。
#
# 模型八的数据：
# 2D数据：
# 1，官方的10min 频谱图（400，300），resize为（300，400）。
# 2，将官方的50s脑电波数据分为4个电极区（与上述相同），并分别转为频谱图（267，501，4），纵向拼接得到（501，1068），resize为（400，800）。
# 3，提取2中的中心官方脑电波数据的中心10s，同等方法转为频谱图（100，291，4），纵向拼接得到（291，400），resize为（300，400）。
# 4，合并1、2、3的频谱图数据（（300，400）、（300，400）、（400，800））得到zoom spec。具体为三部分拼接得到（800，700），resize为（512，512）。
# 5，将上述提取的官方的spec（128，256，4）、频谱转换的梅尔频谱eeg to spec（128，256，4）、4中合成的zoom spec（512，512）、频谱转换的scipy.signal模块的傅里叶变换得到eeg to spec_2（512，512）。
# 6，将其中的spec（128，256，4）与eeg to spec（128，256，4）左右拼接得到（512，512）；将其与zoom spec（512，512）、eeg to spec_2（512，512），转为channel first ，并在第一维度堆叠得到（1，1526，512），第0维度堆叠3次得到2D spec（3，1536，512）。
#
# 1D数据：
# 1，官方的eeg数据（1000，20）分为4个区域，每个区域相邻电极做差。
# 2，对数据滤波处理得到1D eeg（10000，8）即为50s的8个特征的脑电波数据。
import os
import random
from glob import glob

import librosa
import pandas as pd
import pywt
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')
import cupy as cp
import cusignal
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt, iirnotch
import os
import cv2
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy import signal


# zfg configuration

class config:
    BATCH_SIZE = 1
    backbone_2d = "tf_efficientnet_b0"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False
    num_channels = 8
    pretrained = False
    augment = False
    image_transform = transforms.Resize((512, 512))


class paths:
    #     MODEL_WEIGHTS = "/kaggle/input/hms-multi-class-image-classification-train/tf_efficientnet_b0_epoch_3.pth"

    TEST_CSV = "../hms-harmful-brain-activity-classification/A_input_1D_2D/train_small.csv"
    TEST_EEGS = "../hms-harmful-brain-activity-classification/A_input_1D_2D/train_eegs_small/"
    TEST_SPECTROGRAMS = "../hms-harmful-brain-activity-classification/A_input_1D_2D/train_spectrograms_small/"


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


def spectrogram_from_eeg(parquet_path, display=False):  # zfg eeg to spec(3)
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
    """
    Source: https://www.kaggle.com/code/mvvppp/hms-eda-and-domain-journey
    Visualize spectrogram recordings from a parquet file.
    :param spectrogram_path: path to the spectrogram parquet.
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


def create_spectrogram_with_cusignal(eeg_data, eeg_id, start, duration=50,  # zfg 隶属于(4) 完成10s 与 50s
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


def create_spectogram_competition(spec_id, seconds_min):  # zfg 隶属于(5) 完成10min
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


def create_512_spec(eeg_data, eeg_id, spec_id, start_second=0, seconds_min=0):  # zfg kaggle zoom spec(5)
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


def create_spectrogram(data):  # zfg eeg to spec (5)
    """This function will create the spectrograms based on the EEG data with the 'magic formula'."""
    nperseg = 150  # Length of each segment
    noverlap = 128  # Overlap between segments
    NFFT = max(256, 2 ** int(np.ceil(np.log2(nperseg))))

    # Convert data to numpy array if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values

    # LL Spec = ( spec(Fp1 - F7) + spec(F7 - T3) + spec(T3 - T5) + spec(T5 - O1) )/4
    freqs, t, spectrum_LL1 = signal.spectrogram(data[:, feature_to_index['Fp1']] - data[:, feature_to_index['F7']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL2 = signal.spectrogram(data[:, feature_to_index['F7']] - data[:, feature_to_index['T3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL3 = signal.spectrogram(data[:, feature_to_index['T3']] - data[:, feature_to_index['T5']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LL4 = signal.spectrogram(data[:, feature_to_index['T5']] - data[:, feature_to_index['O1']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    LL = (spectrum_LL1 + spectrum_LL2 + spectrum_LL3 + spectrum_LL4) / 4
    # LP Spec = ( spec(Fp1 - F3) + spec(F3 - C3) + spec(C3 - P3) + spec(P3 - O1) )/4
    freqs, t, spectrum_LP1 = signal.spectrogram(data[:, feature_to_index['Fp1']] - data[:, feature_to_index['F3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP2 = signal.spectrogram(data[:, feature_to_index['F3']] - data[:, feature_to_index['C3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP3 = signal.spectrogram(data[:, feature_to_index['C3']] - data[:, feature_to_index['P3']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_LP4 = signal.spectrogram(data[:, feature_to_index['P3']] - data[:, feature_to_index['O1']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    LP = (spectrum_LP1 + spectrum_LP2 + spectrum_LP3 + spectrum_LP4) / 4
    # RP Spec = ( spec(Fp2 - F4) + spec(F4 - C4) + spec(C4 - P4) + spec(P4 - O2) )/4
    freqs, t, spectrum_RP1 = signal.spectrogram(data[:, feature_to_index['Fp2']] - data[:, feature_to_index['F4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP2 = signal.spectrogram(data[:, feature_to_index['F4']] - data[:, feature_to_index['C4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP3 = signal.spectrogram(data[:, feature_to_index['C4']] - data[:, feature_to_index['P4']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    freqs, t, spectrum_RP4 = signal.spectrogram(data[:, feature_to_index['P4']] - data[:, feature_to_index['O2']],
                                                nfft=NFFT, noverlap=noverlap, nperseg=nperseg)
    RP = (spectrum_RP1 + spectrum_RP2 + spectrum_RP3 + spectrum_RP4) / 4
    # RL Spec = ( spec(Fp2 - F8) + spec(F8 - T4) + spec(T4 - T6) + spec(T6 - O2) )/4
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


label_to_num = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
num_to_label = {v: k for k, v in label_to_num.items()}
seed_everything(config.SEED)


def eeg_from_parquet(parquet_path: str) -> np.ndarray:
    """
    This function reads a parquet file and extracts the middle 50 seconds of readings. Then it fills NaN values
    with the mean value (ignoring NaNs).
    :param parquet_path: path to parquet file.
    :param display: whether to display EEG plots or not.
    :return data: np.array of shape  (time_steps, eeg_features) -> (10_000, 8)
    """
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
label_cols = test_df.columns[-6:]
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

for i, eeg_id in tqdm(enumerate(eeg_ids)):  # zfg eeg(1) 数据提取
    # Save EEG to Python dictionary of numpy arrays
    eeg_path = paths.TEST_EEGS + str(eeg_id) + ".parquet"
    data = eeg_from_parquet(eeg_path)
    all_eegs[eeg_id] = data

# zfg read spectrograms

paths_spectrograms = glob(paths.TEST_SPECTROGRAMS + "*.parquet")
print(f'There are {len(paths_spectrograms)} spectrogram parquets')
all_spectrograms = {}

for file_path in tqdm(paths_spectrograms):  # zfg kaggle spec(2) 数据提取
    aux = pd.read_parquet(file_path)
    name = int(file_path.split("/")[-1].split('.')[0])
    all_spectrograms[name] = aux.iloc[:, 1:].values

    del aux

if config.VISUALIZE:
    idx = np.random.randint(0, len(paths_spectrograms))
    spectrogram_path = paths_spectrograms[idx]
    plot_spectrogram(spectrogram_path)

# zfg read eeg spectograms

paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
print(f'There are {len(paths_eegs)} EEG spectrograms')

counter = 0

all_eegs_specs = {}
for file_path in tqdm(paths_eegs):  # zfg eeg to spec 数据提取
    eeg_id = file_path.split("/")[-1].split(".")[0]
    eeg_spectrogram = spectrogram_from_eeg(file_path, counter < 1)
    all_eegs_specs[int(eeg_id)] = eeg_spectrogram  # zfg (128,256,4)
    counter += 1
all_eegs_512_spec = {}
for i in range(len(test_df)):  # zfg kaggle zoom spec 数据提取
    row = test_df.iloc[i]
    eeg_id = row['eeg_id']
    eeg_data = pd.read_parquet(paths.TEST_EEGS + f'{eeg_id}.parquet')
    all_eegs_512_spec[str(eeg_id)] = create_512_spec(eeg_data, eeg_id, row['spectrogram_id'])  # zfg (512,512)
all_eegs_spec_2 = {}
for i in range(len(test_df)):  # zfg eeg to spec 数据提取
    row = test_df.iloc[i]
    eeg_id = row['eeg_id']
    eeg_data = pd.read_parquet(paths.TEST_EEGS + f'{eeg_id}.parquet')
    all_eegs_spec_2[str(eeg_id)] = create_spectrogram(eeg_data)  # create_spectrogram(data)


def print_dict_info(dict_obj, dict_name):
    print(f"Dictionary: {dict_name}")
    for key, value in dict_obj.items():
        print(
            f"Key: {key}, Value Type: {type(value)}, Value Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")


# # 打印 all_eegs_specs 字典的信息
# print_dict_info(all_eegs_specs, 'all_eegs_specs')

# 打印 all_eegs_512_spec 字典的信息
# print_dict_info(all_eegs_512_spec, 'all_eegs_512_spec')
#
# 打印 all_eegs_spec_2 字典的信息
print_dict_info(all_eegs_specs, 'all_eegs_spec_2')
print_dict_info(all_spectrograms, 'all_spectrograms')

# zfg 保存为kaggle spec 和 eeg specs 保存为 npy

np.save('../hms-harmful-brain-activity-classification/A_output_1D_2D/all_eegs_specs.npy', all_eegs_specs,
        allow_pickle=True)
np.save('../hms-harmful-brain-activity-classification/A_output_1D_2D/all_spectrograms.npy', all_spectrograms,
        allow_pickle=True)

#         data = all_eegs[eeg_id]
#         data = create_spectrogram(data)

# plt.figure(figsize=(10, 10))
# plt.imshow(all_eegs_512_spec['3911565283'], cmap='jet')
# plt.axis('off')
# plt.show()

# # unique
# data_mean = 0.00011579196009429602
# data_std = 4.5827806440634316
#
# #zfg dataset
#
#
# from scipy.signal import butter, lfilter
#
# def butter_lowpass_filter(data, cutoff_freq: int = 20, sampling_rate: int = 200, order: int = 4):
#     nyquist = 0.5 * sampling_rate
#     normal_cutoff = cutoff_freq / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     filtered_data = lfilter(b, a, data, axis=0)
#     return filtered_data
data_mean = 0.00011579196009429602
data_std = 4.5827806440634316

# class CustomDataset(Dataset):
#     def __init__(
#             self, df, config, mode='train', specs=all_spectrograms, eeg_specs=all_eegs_specs, downsample=5,
#             augment=config.augment, data_mean=data_mean, data_std=data_std
#     ):
#         self.df = df
#         self.config = config
#         self.mode = mode
#         self.spectrograms = specs
#         self.eeg_spectrograms = eeg_specs
#         self.downsample = downsample
#         self.augment = augment
#         self.USE_KAGGLE_SPECTROGRAMS = True
#         self.USE_EEG_SPECTROGRAMS = True
#         self.all_eegs_512_spec = all_eegs_512_spec
#         self.data_mean = data_mean
#         self.data_std = data_std
#
#     def __len__(self):
#         """
#         Length of dataset.
#         """
#         return len(self.df)
#
#     def __getitem__(self, index):
#         """
#         Get one item.
#         """
#         X1, X2, y = self.__data_generation(index)
#
#         output = {
#             "X1": torch.tensor(X1, dtype=torch.float32),
#             "X2": torch.tensor(X2, dtype=torch.float32),
#             "y": torch.tensor(y, dtype=torch.float32)
#         }
#         return output
#
#     def get_datasetwidenorm(self, eeg_id):
#         """This function will get the batch and preprocess it."""
#         # Set a small epsilon to avoid division by zero
#         eps = 1e-6
#
#         # Read data from parquet file
#         data = all_eegs[eeg_id]
#         data = create_spectrogram(data) #zfg eeg to spec 数据提取
#
#         # Fill missing values with the specified constant
#         mask = np.isnan(data)
#         data[mask] = -1
#
#         # Clip values and apply logarithmic transformation
#         data = np.clip(data, np.exp(-6), np.exp(10))
#         data = np.log(data)
#
#         # Normalize the data
#         data = (data - self.data_mean) / (self.data_std + eps)
#
#         # Convert data to a PyTorch tensor and apply transformations
#         data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
#         data = config.image_transform(data_tensor)
#
#         # Return the batch data
#         return data.numpy()
#
#     def __data_generation(self, index):
#         row = self.df.iloc[index]
#
#         X1, X2 = self.__spec_data_gen(row)
#
#         y = np.zeros(6, dtype='float32')
#         if self.mode != 'test':
#             y = row[label_cols].values.astype(np.float32)
#
#         return X1, X2, y
#
#     def __spec_data_gen(self, row):
#         """
#         Generates spec data containing batch_size samples.
#         """
#         spec = self.all_eegs_512_spec[str(row['eeg_id'])]
#         spec = spec[np.newaxis, :, :]
#         spec = np.tile(spec, (3, 1, 1))
#
#         new_norm = self.get_datasetwidenorm(row['eeg_id'])
#         new_norm = np.tile(new_norm, (3, 1, 1))
#
#         return spec, new_norm


# zfg dataloader

# test_dataset = CustomDataset(test_df, config, mode="test")
# test_loader = DataLoader(
#     test_dataset,
#     batch_size=1,
#     shuffle=False,
#     num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False
# )
# output = test_dataset[0]
# X1, X2, y = output["X1"], output["X2"], output["y"]
#
# print(f"X1 shape: {X1.shape}")
# print(f"X2 shape: {X2.shape}")
# print(f"y shape: {y.shape}")
