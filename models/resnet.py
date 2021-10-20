import torch
import pandas as pd
from objects import Model
from .utils import prune_df, get_all_shape
from torchvision.models import resnet18, vgg16, alexnet, mobilenet_v3_large


def get_resnet(gpu_list):
    """Return a Resnet model."""
    max_batch = 3
    resnet_layer = dict()
    for i in range(1, max_batch + 1):
        for gpu in gpu_list:
            filename = f"data/Resnet/ResNet_{gpu.csv_name}_{i}.csv"
            if not i in resnet_layer:
                resnet_layer[i] = dict()
            resnet_layer[i][gpu.id] = prune_df(
                pd.read_csv(filename))['duration']

    ResNet = Model("ResNet", get_all_shape(resnet18()), resnet_layer)
    return ResNet
