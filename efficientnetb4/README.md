---
license: apache-2.0
tags:
- vision
- image-classification
datasets:
- imagenet-1k
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace
---

# EfficientNet (b4 model) 

EfficientNet model trained on ImageNet-1k at resolution 380x380. It was introduced in the paper [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
](https://arxiv.org/abs/1905.11946) by Mingxing Tan and Quoc V. Le, and first released in [this repository](https://github.com/keras-team/keras). 

Disclaimer: The team releasing EfficientNet did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

EfficientNet is a mobile friendly pure convolutional model (ConvNet) that proposes a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient.

![model image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/efficientnet_architecture.png)

## Intended uses & limitations

You can use the raw model for image classification. See the [model hub](https://huggingface.co/models?search=efficientnet) to look for
fine-tuned versions on a task that interests you.

### How to use

Here is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:

```python
import torch
from datasets import load_dataset
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

preprocessor = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b4")
model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b4")

inputs = preprocessor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label]),
```

For more code examples, we refer to the [documentation](https://huggingface.co/docs/transformers/master/en/model_doc/efficientnet).

### BibTeX entry and citation info

```bibtex
@article{Tan2019EfficientNetRM,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Mingxing Tan and Quoc V. Le},
  journal={ArXiv},
  year={2019},
  volume={abs/1905.11946}
}
```