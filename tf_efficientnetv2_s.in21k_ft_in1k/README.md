---
tags:
- image-classification
- timm
library_name: timm
license: apache-2.0
datasets:
- imagenet-1k
- imagenet-21k
---
# Model card for tf_efficientnetv2_s.in21k_ft_in1k

A EfficientNet-v2 image classification model. Trained on ImageNet-21k and fine-tuned on ImageNet-1k in Tensorflow by paper authors, ported to PyTorch by Ross Wightman.


## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 21.5
  - GMACs: 5.4
  - Activations (M): 22.7
  - Image size: train = 300 x 300, test = 384 x 384
- **Papers:**
  - EfficientNetV2: Smaller Models and Faster Training: https://arxiv.org/abs/2104.00298
- **Dataset:** ImageNet-1k
- **Pretrain Dataset:** ImageNet-21k
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

model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=True)
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
    'tf_efficientnetv2_s.in21k_ft_in1k',
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
    #  torch.Size([1, 24, 150, 150])
    #  torch.Size([1, 48, 75, 75])
    #  torch.Size([1, 64, 38, 38])
    #  torch.Size([1, 160, 19, 19])
    #  torch.Size([1, 256, 10, 10])

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
    'tf_efficientnetv2_s.in21k_ft_in1k',
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
# output is unpooled, a (1, 1280, 10, 10) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
Explore the dataset and runtime metrics of this model in timm [model results](https://github.com/huggingface/pytorch-image-models/tree/main/results).

## Citation
```bibtex
@inproceedings{tan2021efficientnetv2,
  title={Efficientnetv2: Smaller models and faster training},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International conference on machine learning},
  pages={10096--10106},
  year={2021},
  organization={PMLR}
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
