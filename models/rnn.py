import torch
import pandas as pd
from objects import Model
from .utils import prune_df, get_all_shape
from torchvision.models import resnet18, vgg16, alexnet, mobilenet_v3_large


def get_rnn(gpu_list):
    """Return a RNN model."""
    max_batch = 3
    rnn_layer = dict()
    for i in range(1, max_batch + 1):
        for gpu in gpu_list:
            filename = f"data/RNN/rnn_{gpu.csv_name}_{i}.csv"
            if not i in rnn_layer:
                rnn_layer[i] = dict()
            rnn_layer[i][gpu.id] = prune_df(pd.read_csv(filename))['duration']

    rnn_shape = [
        torch.Size([1, 28, 28]),
        torch.Size([1, 28, 28]),
        torch.Size([1, 128])
    ]
    return Model("RNN", rnn_shape, rnn_layer)
