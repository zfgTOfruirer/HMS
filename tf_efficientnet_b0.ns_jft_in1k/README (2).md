---
tags:
- image-classification
- timm
library_name: timm
license: apache-2.0
datasets:
- imagenet-1k
---
# Model card for tf_efficientnet_b0.ns_jft_in1k

A EfficientNet image classification model. Trained on ImageNet-1k and unlabeled JFT-300m using Noisy Student semi-supervised learning in Tensorflow by paper authors, ported to PyTorch by Ross Wightman.


## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 5.3
  - GMACs: 0.4
  - Activations (M): 6.7
  - Image size: 224 x 224
- **Papers:**
  - EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks: https://arxiv.org/abs/1905.11946
  - Self-training with Noisy Student improves ImageNet classification: https://arxiv.org/abs/1911.04252
- **Dataset:** ImageNet-1k
- **Original:** https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('tf_efficientnet_b0.ns_jft_in1k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

### Feature Map Extraction
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'tf_efficientnet_b0.ns_jft_in1k',
    pretrained=True,
    features_only=True,
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

for o in output:
    # print shape of each feature map in output
    # e.g.:
    #  torch.Size([1, 16, 112, 112])
    #  torch.Size([1, 24, 56, 56])
    #  torch.Size([1, 40, 28, 28])
    #  torch.Size([1, 112, 14, 14])
    #  torch.Size([1, 320, 7, 7])

    print(o.shape)
```

### Image Embeddings
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'tf_efficientnet_b0.ns_jft_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 1280, 7, 7) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
Explore the dataset and runtime metrics of this model in timm [model results](https://github.com/huggingface/pytorch-image-models/tree/main/results).

## Citation
```bibtex
@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International conference on machine learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}
```
```bibtex
@article{Xie2019SelfTrainingWN,
  title={Self-Training With Noisy Student Improves ImageNet Classification},
  author={Qizhe Xie and Eduard H. Hovy and Minh-Thang Luong and Quoc V. Le},
  journal={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019},
  pages={10684-10695}
}
```
```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/huggingface/pytorch-image-models}}
}
```
