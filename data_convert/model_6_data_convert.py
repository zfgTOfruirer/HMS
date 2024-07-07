# zfg 模型六的数据：

# 1，采用与模型三、四、五相同的数据转换方式。
# 2，提取官方eeg 50S的脑电波数据进行转换得到（516，448）大小的频谱图。
# 3，提取官方eeg 中心10S的脑电波数据进行转换得到（516，108）大小的频谱图。
# 4，左右拼接得到（516，533），并resize为（512，512）。


import os
import typing as tp
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT = Path("../hms-harmful-brain-activity-classification")
INPUT = ROOT / "input_6"
OUTPUT = ROOT / "output_6"
DATA = INPUT / "train-raw-csv"
TEST_SPEC = DATA / "train_spectrograms_small_6"
TMP = ROOT / "tmp_6"
TEST_SPEC_SPLIT = TMP / "train_spectrograms_split_6"
TMP.mkdir(exist_ok=True)
TEST_SPEC_SPLIT.mkdir(exist_ok=True)

RANDAM_SEED = 1086
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
N_CLASSES = len(CLASSES)

test = pd.read_csv("../hms-harmful-brain-activity-classification/output_6/train_small.csv")

# RENAME FOR DATALOADER
test['spec_id'] = test['spectrogram_id']
# test = test.rename({'spectrogram_id':'spec_id'},axis=1)

eeg_features = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4",
                "T6", "O2", "EKG"]

feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}

import pywt
from scipy import signal

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
    data = eeg.values[:10000, :]
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
PATH2 = '../hms-harmful-brain-activity-classification/input_6/train-raw-csv/train_eegs_small_6/'
DISPLAY = 1
EEG_IDS2 = test.eeg_id.unique()
all_eegs = {}

print('Converting Test EEG to Spectrograms...');
print()
for i, eeg_id in enumerate(EEG_IDS2):
    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f'{PATH2}{eeg_id}.parquet', i < DISPLAY, type=1)  # type 1 为默认的50s
    all_eegs[eeg_id] = img

all_eegs1 = {}
for i, eeg_id in enumerate(EEG_IDS2):
    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f'{PATH2}{eeg_id}.parquet', i < DISPLAY, type=2)  # type 2  抽取中间的10s
    all_eegs1[eeg_id] = img

FilePath = tp.Union[str, Path]
EegIds = tp.Union[str, Path]
# Label = tp.Union[int, float, np.ndarray]
LabelIds = tp.Union[str, Path]


class HMSHBACSpecDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            eeg_ids: tp.Sequence[EegIds],
            image_paths: tp.Sequence[FilePath],
            label_ids: tp.Sequence[LabelIds],
            transform: A.Compose,
    ):
        self.eeg_ids = eeg_ids
        self.image_paths = image_paths
        self.label_ids = label_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        eeg_id = self.eeg_ids[index]
        label_id = self.label_ids[index]
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

        return {"data": img, "target": label_id}

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img


class CFG:
    img_size = 512


def get_test_path_label(test: pd.DataFrame):
    """Get file path and dummy target info."""
    img_paths = []
    eeg_ids = []
    label_ids = []

    for spec_id in test["spectrogram_id"].values:
        img_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"  # 确保TEST_SPEC_SPLIT已定义为Path类型
        img_paths.append(img_path)

        # 获取当前spectrogram_id的行
        test_row = test[test['spectrogram_id'] == spec_id]

        # 获取对应的eeg_id
        eeg_id = test_row['eeg_id'].values[0]
        eeg_ids.append(eeg_id)

        # 获取对应的label_id
        label_id = test_row['label_id'].values[0]  # 假设label_id是DataFrame中的一列
        label_ids.append(label_id)

    test_data = {
        "eeg_ids": eeg_ids,
        "image_paths": img_paths,
        "label_ids": label_ids
    }

    return test_data


def get_test_transforms(CFG):
    test_transform = A.Compose([
        A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
        ToTensorV2(p=1.0)
    ])
    return test_transform


test_path_label = get_test_path_label(test)
test_transform = get_test_transforms(CFG)
test_dataset = HMSHBACSpecDataset(**test_path_label, transform=test_transform)

# zfg 保存图像
import os
import numpy as np
from PIL import Image

# 确保输出目录存在
output_dir = OUTPUT / "A"
os.makedirs(output_dir, exist_ok=True)

# 准备 DataLoader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# 保存图像
for data in test_loader:
    img = data['data']
    print(img.shape)  # 加载图像
    print(img)
    label_id = data['target']  # 加载对应的 label_id
    img_name = f"{label_id.item()}.tiff"  # 构建文件名，使用 TIFF 格式
    img_path = os.path.join(output_dir, img_name)
    img = img.squeeze()
    img = (img.numpy() * 255).astype(np.float32)  # 转换为 float32 并且进行缩放
    img_pil = Image.fromarray(img)
    img_pil.save(img_path, format='TIFF')  # 保存图像为 TIFF 格式

# 准备一个空字典来存储数据
image_dict = {}

# 读取文件夹中的所有图像
for filename in os.listdir(output_dir):
    if filename.endswith('.tiff'):
        image_path = os.path.join(output_dir, filename)
        img = np.array(Image.open(image_path)).astype(np.float32)  # 读取时确保转换为 float32
        key = os.path.splitext(filename)[0]  # 去除文件后缀来作为键
        image_dict[int(key)] = img  # 将图像数据存入字典

        # 打印每个键和对应图像的基本信息
        print("Key: '{}' - Image shape: {}, dtype: {}".format(key, img.shape, img.dtype))

# 序列化并保存字典
np.save(os.path.join(output_dir, 'combine_115.npy'), image_dict, allow_pickle=True)

# 打印成功存储的图像数量
print("Successfully saved {} images to combine_115.npy".format(len(image_dict)))
