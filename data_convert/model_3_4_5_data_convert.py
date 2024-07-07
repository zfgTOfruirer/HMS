# zfg 模型三、四、五的数据：

# 1，主要将官方的脑电波数据转为频谱图数据（eeg to spec），转换处理的流程与模型一的1部分类似。
# 2，将各电极的脑电波数据分为4个区域（因为官方的频谱图也为纵向4个区域对应4个电极区），每个区域内若干脑电波数据相邻做差，得到的每个区域含有的4个脑电波。
# 3，4个区域，每个区域4个脑电波数据通过scipy.signal模块的傅里叶变换完成转换。每个区域的频谱为该区域的4个频谱的平均频谱。最终纵向拼接得到（512，512）大小的频谱图。

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
INPUT = ROOT / "input_3_4_5"
OUTPUT = ROOT / "output_3_4_5"
DATA = INPUT / "train-raw-csv"
TEST_SPEC = DATA / "train_spectrograms_small_3_4_5"
TMP = ROOT / "tmp_3_4_5"
TEST_SPEC_SPLIT = TMP / "train_spectrograms_split_3_4_5"
TMP.mkdir(exist_ok=True)
TEST_SPEC_SPLIT.mkdir(exist_ok=True)

RANDAM_SEED = 1086
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
N_CLASSES = len(CLASSES)

test = pd.read_csv("../hms-harmful-brain-activity-classification/output_3_4_5/train_small.csv")

for spec_id in test["spectrogram_id"]:
    spec = pd.read_parquet(TEST_SPEC / f"{spec_id}.parquet")

    spec_arr = spec.fillna(0).values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)

    np.save(TEST_SPEC_SPLIT / f"{spec_id}.npy", spec_arr)

# # READ ALL SPECTROGRAMS
# PATH2 = '/media/laplace/CA4960E6F5A4B1BD/hms-harmful-brain-activity-classification/input_3_4_5/test_spectrograms/'
# files2 = os.listdir(PATH2)
# print(f'There are {len(files2)} test spectrogram parquets')
#
# spectrograms2 = {}
# for i, f in enumerate(files2):
#     if i % 100 == 0: print(i, ', ', end='')
#     tmp = pd.read_parquet(f'{PATH2}{f}')
#     name = int(f.split('.')[0])
#     spectrograms2[name] = tmp.iloc[:, 1:].values

# RENAME FOR DATALOADER
test['spec_id'] = test['spectrogram_id']

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
PATH2 = '../hms-harmful-brain-activity-classification/input_3_4_5/train-raw-csv/train_eegs_small_3_4_5/'
DISPLAY = 1
EEG_IDS2 = test.eeg_id.unique()
all_eegs = {}

print('Converting Test EEG to Spectrograms...');
print()
for i, eeg_id in enumerate(EEG_IDS2):
    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f'{PATH2}{eeg_id}.parquet', i < DISPLAY)
    all_eegs[eeg_id] = img

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
