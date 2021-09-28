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

    def assign(self, task):
        print(
            f"[Log] Assign layer {task.current_layer} of task {task.model.name} to device {self.name}"
        )
        model = task.model
        layer_latency = model.layer_latency[1][self.GPUSpec.id][
            task.current_layer]
        self.remain_time = layer_latency
        self.task = task


class Model:
    def __init__(self, name, layer_input_shape, layer_latency):
        self.id = uuid.uuid4()
        self.name = name
        self.layer_input_shape = layer_input_shape
        self.layer_latency = layer_latency
        self.layer_movement_time = self.get_movement_time(
            self.layer_input_shape)

    @staticmethod
    def get_movement_time(layer_input_shape):
        movement_time = list()

        for shape in layer_input_shape:
            sleep(0.1)
            data = torch.randn(shape)
            data.to(0)
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                data.to(1)
            movement_time.append(round(prof.self_cpu_time_total * 1000))
        movement_time.append(0)
        return movement_time


class Task:
    def __init__(self, model, current_layer=0, input_data_position=0):
        self.id = uuid.uuid4()
        self.model = model
        self.current_layer = current_layer
        self.input_date_position = input_data_position
