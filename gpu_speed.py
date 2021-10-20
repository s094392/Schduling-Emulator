import math
from copy import deepcopy
from tqdm import tqdm
import random
import logging
import pandas as pd
from objects import DeviceType, GPUSpec, Device, Task
from models.resnet import get_resnet
from models.alexnet import get_alexnet
from models.rnn import get_rnn
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def nextTime(rateParameter):
    return -math.log(1.0 - random.random()) / rateParameter

def emulate(Tasks, Devices, schedule):
    Queue = []
    DoneTasks = []

    tail_latency = 0
    while True:

        # find nearest device
        min_delay = float("inf")
        for device in Devices:
            if device.type == DeviceType.GPU:
                if device.task:
                    min_delay = min((min_delay, device.remain_time))
        if len(Tasks):
            min_delay = min((min_delay, Tasks[0].model.size))

        # update remain time of each deivce and assign next layer if avaliable
        for device in Devices:
            if device.type == DeviceType.GPU:
                if device.task:
                    device.remain_time -= min_delay
                    if device.remain_time == 0:
                        if device.task.current_layer + 1 < len(
                                device.task.model.layer_input_shape):
                            device.task.current_layer += 1
                            device.assign(device.task)
                        else:
                            device.task = None

        if min_delay == float("inf"):
            if len(Tasks) == 0 and len(Queue) == 0:
                break
            if len(Tasks):
                tail_latency = Tasks[0].arrival_time
        else:
            tail_latency += min_delay

        # update queue
        while len(Tasks):
            if Tasks[0].arrival_time <= tail_latency:
                task = Tasks.pop(0)
                Queue.append(task)
                DoneTasks.append(task)
            else:
                break

        for device in Devices:
            if device.type == DeviceType.GPU:
                if not device.task and len(Queue):
                    logging.info(
                        f"[Log] {tail_latency}"
                    )
                    schedule(Queue, device, tail_latency)

    model_latencies = {}
    Tasks.extend(DoneTasks)
    return tail_latency


def fifo_schedule(Queue, device, tail_latency):
    task = Queue.pop(0)
    task.schedule_time = tail_latency
    device.assign(task)


def sjf_schedule(Queue, device, tail_latency):
    Queue.sort()
    task = Queue.pop(0)
    task.schedule_time = tail_latency
    device.assign(task)

def nonavie_schedule(Queue, device, tail_latency):
    Queue.sort()
    if device.name == "GPU (2060)":
        task = Queue.pop(len(Queue) - 1)
        task.schedule_time = tail_latency
        device.assign(task)
    else:
        task = Queue.pop(0)
        task.schedule_time = tail_latency
        device.assign(task)

def navie_schedule(Queue, device, tail_latency):
    Queue.sort()
    if device.name == "GPU (2060)":
        task = Queue.pop(0)
        task.schedule_time = tail_latency
        device.assign(task)
    else:
        task = Queue.pop(len(Queue) - 1)
        task.schedule_time = tail_latency
        device.assign(task)

def main():
    logging.basicConfig(filename='info.log',
                        encoding='utf-8',
                        level=logging.INFO)

    GPU_2060 = GPUSpec("RTX 2060", "2060")
    GPU_1080 = GPUSpec("GTX 1008 Ti", "1080")
    gpu_list = (GPU_2060, GPU_1080)

    Tasks = []
    Devices = []
    Models = []

    ResNet = get_resnet(gpu_list)
    AlexNet = get_alexnet(gpu_list)
    RNN = get_rnn(gpu_list)

    Models.append(ResNet)
    Models.append(AlexNet)
    Models.append(RNN)
    Devices.append(Device("GPU", device_type=DeviceType.CPU, device_id=0))

    GPU_0 = Device("GPU (2060)", device_type=DeviceType.GPU, GPUSpec=GPU_2060)
    GPU_1 = Device("GPU (1080)", device_type=DeviceType.GPU, GPUSpec=GPU_1080)
    Devices.append(GPU_0)
    Devices.append(GPU_1)

    for rate in (1/10000, 1/500000, 1/100000, 1/1000000, 1/5000000, 1/10000000, 1/100000000):
        test_data = []

        N = 5
        tasks_size = 100
        test_datas = []
        for i in range(N):
            now = 0
            for _ in range(tasks_size):
                now+=nextTime(rate)
                test_data.append(Task(random.choice((ResNet, AlexNet, RNN)), arrival_time=now))
            test_data = sorted(test_data, key=lambda x: x.arrival_time)
            test_datas.append(test_data)

        results = []
        for j in tqdm(range(N)):
            test_data = test_datas[j]
            Tasks = deepcopy(test_data)
            tail_latency = emulate(Tasks, Devices, navie_schedule)
            results.append(tail_latency)
            # results.append(sum([task.schedule_time - task.arrival_time for task in Tasks]))
        df = pd.DataFrame(results)
        # print("Navie result:")
        # print(df.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))))
        navie_mean = sum(results) / len(results)

        results = []
        for j in tqdm(range(N)):
            test_data = test_datas[j]
            Tasks = deepcopy(test_data)
            tail_latency = emulate(Tasks, Devices, nonavie_schedule)
            results.append(tail_latency)
            # results.append(sum([task.schedule_time - task.arrival_time for task in Tasks]))
        df = pd.DataFrame(results)
        # print("Navie result:")
        # print(df.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))))
        nonavie_mean = sum(results) / len(results)

        results = []
        for j in tqdm(range(N)):
            test_data = test_datas[j]
            Tasks = deepcopy(test_data)
            Tasks = sorted(Tasks, key=lambda x: x.arrival_time)
            tail_latency = emulate(Tasks, Devices, fifo_schedule)
            results.append(tail_latency)
            # results.append(sum([task.schedule_time - task.arrival_time for task in Tasks]))
        df = pd.DataFrame(results)
        # print("FIFO result:")
        # print(df.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))))
        fifo_mean = sum(results) / len(results)

        results = []
        for j in tqdm(range(N)):
            test_data = test_datas[j]
            Tasks = deepcopy(test_data)
            Tasks = sorted(Tasks, key=lambda x: x.arrival_time)
            tail_latency = emulate(Tasks, Devices, sjf_schedule)
            results.append(tail_latency)
            # results.append(sum([task.schedule_time - task.arrival_time for task in Tasks]))
        df = pd.DataFrame(results)
        # print("SJF result:")
        # print(df.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))))
        sjf_mean = sum(results) / len(results)


        print(1/rate, navie_mean / fifo_mean, navie_mean / sjf_mean, navie_mean / nonavie_mean)


if __name__ == "__main__":
    # print(nextTime(1/100000))
    main()
