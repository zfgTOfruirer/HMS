# zfg 模型一、二的数据：

# 1，官方的每条50s脑电波数据转为频谱图（raw eeg -> eeg to spec）将各电极的脑电波数据分为4个区域（因为官方的频谱图也为纵向4个区域对应4个电极区），
# 每个区域脑电波数据相邻做差；4个区域的脑电波数据再通过librosa转为4个梅尔频谱图，尺寸为（128，256，4）（其他的也尝试过，针对此处转为梅尔频谱图，效果更好），
# 纵向拼接得到最终频谱图eeg to spec尺寸为（512，256）。
# 2，官方的每条10min频谱图数据
# 将官方频谱图像kaggle spec对数归一化，分割为（100，300，4），每行间隔14个像素的距离，纵向拼接，并resize为（512，256）。
# 3，将1与2横向拼接组合为（512，512）大小的组合频谱图，此时频谱图从上至下分别为四个电极区对应的频谱。

import os
import typing as tp
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT = Path("../hms-harmful-brain-activity-classification")
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
DATA = INPUT / "train-raw-csv"
TEST_SPEC = DATA / "train_spectrograms_small"
TMP = ROOT / "tmp"
TEST_SPEC_SPLIT = TMP / "train_spectrograms_split"
TMP.mkdir(exist_ok=True)
TEST_SPEC_SPLIT.mkdir(exist_ok=True)

RANDAM_SEED = 1086
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
N_CLASSES = len(CLASSES)
# FOLDS = [0,1,2,3,4,5,6,7,8,9,10]
# N_FOLDS = len(FOLDS)
test = pd.read_csv("../hms-harmful-brain-activity-classification/output/train_small.csv")

# zfg 下面为 eeg to spec

#
# for spec_id in test["spectrogram_id"]:
#     spec = pd.read_parquet(TEST_SPEC / f"{spec_id}.parquet")
#     spec_arr = spec.fillna(0).values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)
#     np.save(TEST_SPEC_SPLIT / f"{spec_id}.npy", spec_arr)
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

im = np.load("../hms-harmful-brain-activity-classification/tmp/train_spectrograms_split/67817419.npy")
print(im.shape)

test = pd.read_csv('../hms-harmful-brain-activity-classification/output/train_small.csv')
print('Test shape', test.shape)
import pywt, librosa


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
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128, 256, 4), dtype='float32')

    if display: plt.figure(figsize=(10, 7))
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


# READ ALL EEG SPECTROGRAMS
PATH2 = '../hms-harmful-brain-activity-classification/input/train-raw-csv/train_eegs_small/'
DISPLAY = 1
EEG_IDS2 = test.eeg_id.unique()
all_eegs = {}

print('Converting Test EEG to Spectrograms...');
print()
for i, eeg_id in enumerate(EEG_IDS2):
    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f'{PATH2}{eeg_id}.parquet', i < DISPLAY)
    all_eegs[eeg_id] = img  # zfg eeg 转为 spec

# zfg 下面为拼接（原始+eeg to spec）

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
        # return {"data": img, "target": label}
        return {"data": img, "target": label_id}

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


def get_test_path_label(test: pd.DataFrame):
    """Get file path and target label info."""
    img_paths = []
    eeg_ids = []
    label_ids = []

    # 循环遍历spectrogram_id获取相关信息
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

    # 将收集到的信息组装成字典
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


# test_preds_arr = np.zeros((N_FOLDS, len(test), N_CLASSES))
test_path_label = get_test_path_label(test)
test_transform = get_test_transforms(CFG)
test_dataset = HMSHBACSpecDataset(**test_path_label, transform=test_transform)

# #zfg 保存图像0ld
#
# # 确保输出目录存在
# output_dir = OUTPUT/"A"
# os.makedirs(output_dir, exist_ok=True)
#
# # 准备 DataLoader
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
#
# # 保存图像
# for data in test_loader:
#     img = data['data']
#     print(img.shape)# 加载图像
#     print(img)
#     label_id = data['target']  # 加载对应的 label_id
#     img_name = f"{label_id.item()}.png"  # 假设 label_id 是 tensor，构建文件名
#     img_path = os.path.join(output_dir, img_name)
#     img = img.squeeze()
#     img = Image.fromarray((img.numpy() * 255).astype(np.uint8))  # 转换为 uint8
#     img.save(img_path)  # 保存图像
#
# import os
# import numpy as np
# from PIL import Image
#
# # 准备一个空字典来存储数据
# image_dict = {}
#
# # 读取文件夹中的所有图像
#   # 确保这是你的图像存储文件夹的路径
# for filename in os.listdir(output_dir):
#     if filename.endswith('.png'):
#         image_path = os.path.join(output_dir, filename)
#         img = np.array(Image.open(image_path))
#         key = os.path.splitext(filename)[0]  # 去除文件后缀来作为键
#         image_dict[np.int64(key)] = img  # 将图像数据存入字典
#
#         # 打印每个键和对应图像的基本信息
#         print("Key: '{}' - Image shape: {}, dtype: {}".format(key, img.shape, img.dtype))
#
# # 序列化并保存字典
# # 确保这是你的保存位置路径
# np.save(os.path.join(OUTPUT, 'combine_120.npy'), image_dict, allow_pickle=True)
#
# # 打印成功存储的图像数量
# print("Successfully saved {} images to combine_120.npy".format(len(image_dict)))

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
