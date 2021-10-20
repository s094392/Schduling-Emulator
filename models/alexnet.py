import torch
import pandas as pd
from objects import Model
from .utils import prune_df, get_all_shape
from torchvision.models import resnet18, vgg16, alexnet, mobilenet_v3_large


def get_alexnet(gpu_list):
    """Return a Alexnet model."""
    max_batch = 3
    alexnet_layer = dict()
    for i in range(1, max_batch + 1):
        for gpu in gpu_list:
            filename = f"data/AlexNet/AlexNet_{gpu.csv_name}_{i}.csv"
            if not i in alexnet_layer:
                alexnet_layer[i] = dict()
            alexnet_layer[i][gpu.id] = prune_df(
                pd.read_csv(filename))['duration']

    AlexNet = Model("AlexNet", get_all_shape(alexnet()), alexnet_layer)
    return AlexNet
