import pandas as pd
import numpy as np
import logging
import uuid
import torch

from time import sleep
from enum import Enum


class DeviceType(Enum):
    CPU = 0
    GPU = 1


class GPUSpec:
    def __init__(self, name, csv_name):
        self.id = uuid.uuid4()
        self.name = name
        self.csv_name = csv_name


class Device:
    def __init__(self,
                 name,
                 device_type=DeviceType.GPU,
                 GPUSpec=None,
                 task=None,
                 device_id=uuid.uuid4()):
        self.name = name
        self.GPUSpec = GPUSpec
        self.type = device_type
        self.task = task
        self.remain_time = 0
        self.layer_remain_time = 0

    def forward(self, task):
        logging.info(
            f"[Log] Assign layer {task.current_layer} of task {task.model.name} ({task.arrival_time})to device {self.name}"
        )
        model = task.model
        layer_latency = model.layer_latency[task.batch_size][self.GPUSpec.id][
            task.current_layer]
        self.layer_remain_time = layer_latency
        last_layer_latency = model.layer_latency[task.batch_size][self.GPUSpec.id][
            task.current_layer - 1]
        self.remain_time = self.remain_time - last_layer_latency
        self.task = task

    def assign(self, task):
        logging.info(
            f"[Log] Assign layer {task.current_layer} of task {task.model.name} ({task.arrival_time})to device {self.name}"
        )
        model = task.model
        layer_latency = model.layer_latency[task.batch_size][self.GPUSpec.id][
            task.current_layer]
        self.layer_remain_time = layer_latency
        self.remain_time = sum(model.layer_latency[task.batch_size][self.GPUSpec.id][
                task.current_layer:])
        self.task = task

    def move(self, device):
        logging.info(
            f"[Log] move task {task.model.name} ({task.arrival_time}) from device {self.name} to device {device.name}"
        )
        device.task = self.task
        self.task = None

    def move_and_assign(self, device, task):
        self.move(device)
        self.assign(task)

    def batch_and_assign(self, task):
        assert (task.model == self.task.model)
        current_layer = self.task.current_layer
        model = task.model
        layer_latency = {}
        layer_latency[1] = {}
        for gpu in model.layer_latency[1]:
            layer_latency[1][gpu] = pd.concat([
                model.layer_latency[1][gpu][:current_layer],
                model.layer_latency[2][gpu][current_layer:]], ignore_index=True)
        new_model = Model(f"{task.id} + {self.task.id}",
                          model.layer_input_shape, layer_latency)
        self.task = Task(new_model, arrival_time=0, batch_size=1)


class Model:
    def __init__(self, name, layer_input_shape, layer_latency):
        self.id = uuid.uuid4()
        self.name = name
        self.layer_input_shape = layer_input_shape

        self.layer_latency = layer_latency
        self.layer_movement_time = self.get_movement_time(
            self.layer_input_shape)

        self.gpu_size = {}
        for gpu in layer_latency[1]:
            self.gpu_size[gpu] = sum(layer_latency[1][gpu])

    def adjust_latency(self, GPU, rate):
        """Adjust the latency of specific GPU."""
        for batch_size in self.layer_latency:
            # print(type(self.layer_latency[batch_size][GPU]))
            self.layer_latency[batch_size][GPU] *= rate

    @staticmethod
    def get_movement_time(layer_input_shape):
        movement_time = list()

        for shape in layer_input_shape:
            torch.cuda.synchronize()
            data = torch.randn(shape)
            data.to(0)
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                data.to(1)
            movement_time.append(round(prof.self_cpu_time_total * 1000))
        movement_time.append(0)
        return movement_time


class Task:
    def __init__(self,
                 model,
                 current_layer=0,
                 input_data_position=0,
                 arrival_time=0,
                 batch_size=1):
        self.id = uuid.uuid4()
        self.model = model
        self.current_layer = current_layer
        self.batch_size = batch_size
        self.size = sum([
            sum(i[1])
            for i in self.model.layer_latency[self.batch_size].items()
        ]) / len(self.model.layer_latency[self.batch_size])
        self.input_data_position = input_data_position
        self.arrival_time = arrival_time
        self.schedule_time = 0

    def __lt__(self, other):
        return self.size < other.size
