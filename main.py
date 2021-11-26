#!/usr/bin/env python
# -*- coding: utf-8 -*- #
"""The main.py."""

import math
from copy import deepcopy
from tqdm import tqdm
import random
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from objects import DeviceType, GPUSpec, Device, Task
from models.resnet import get_resnet
from models.alexnet import get_alexnet
from models.rnn import get_rnn
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def nextTime(rateParameter):
    """Poisson distribution generator."""
    return -math.log(1.0 - random.random()) / rateParameter


def emulate(Tasks, Devices, schedule, device_schedule=None):
    """Emulate the tasks distribution based on the scheduler and Return the tail latency."""
    Queue = []
    DoneTasks = []

    tail_latency = 0
    while True:

        # find nearest device
        min_delay = float("inf")
        for device in Devices:
            if device.type == DeviceType.GPU:
                if device.task:
                    min_delay = min((min_delay, device.layer_remain_time))
        if len(Tasks):
            min_delay = min((min_delay, Tasks[0].size))

        # update remain time of each deivce and assign next layer if avaliable
        for device in Devices:
            if device.type == DeviceType.GPU:
                if device.task:
                    device.layer_remain_time -= min_delay
                    if device.layer_remain_time == 0:
                        if device.task.current_layer + 1 < len(
                                device.task.model.layer_input_shape):
                            device.task.current_layer += 1
                            device.forward(device.task)
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
                    logging.info(f"[Log] {tail_latency}")
                    schedule(Devices, Queue, device, tail_latency)

        if device_schedule:
            device_schedule(Devices, Queue)

    model_latencies = {}
    Tasks.extend(DoneTasks)
    return tail_latency


def fifo_schedule(Devices, Queue, device, tail_latency):
    """Fifo scheduler."""
    task = Queue.pop(0)
    task.schedule_time = tail_latency
    device.assign(task)


def sjf_schedule(Devices, Queue, device, tail_latency):
    """Sortest Job First scheduler."""
    Queue.sort()
    task = Queue.pop(0)
    task.schedule_time = tail_latency
    device.assign(task)


def nonaive_schedule(Devices, Queue, device, tail_latency):
    """Distribute the biggest task to the slowest GPU."""
    Queue.sort()
    if device.name == "GPU (2060)":
        task = Queue.pop(len(Queue) - 1)
        task.schedule_time = tail_latency
        device.assign(task)
    else:
        task = Queue.pop(0)
        task.schedule_time = tail_latency
        device.assign(task)

def batch_schedule(Devices, Queue, device, tail_latency):
    """Batch all possiable tasks in job queue."""
    task_pairs = {}

    for job in Queue:
        task_pair = (job.model, job.current_layer)
        max_batch = max(task_pair[0].layer_latency.keys())
        if task_pair not in task_pairs:
            task_pairs[task_pair] = []
        # if job.batch_size < max_batch:
        task_pairs[task_pair].append(job.batch_size)
    
    Queue.clear()

    for task_pair in task_pairs:
        total_batch = sum(task_pairs[task_pair])
        max_batch = max(task_pair[0].layer_latency.keys())
        while(total_batch):
            # print(total_batch)
            batch_size = min(total_batch, max_batch)
            task = Task(task_pair[0], arrival_time=0, batch_size=batch_size)
            total_batch -= task.batch_size
            Queue.append(task)

    naive_schedule(Devices, Queue, device, tail_latency)

def lazy_batching_schedule(Devices, Queue):
    def batched_size(current_layer, device, model):
        return sum(model.layer_latency[1][device.GPUSpec.id][:current_layer]) + sum(model.layer_latency[2][device.GPUSpec.id][current_layer:])
    for device in Devices:
        if device.task and 2 in device.task.model.layer_latency:
            batched_latency = batched_size(device.task.current_layer, device, device.task.model)
            original_latency = sum(device.task.model.layer_latency[1][device.GPUSpec.id][device.task.current_layer:])
            if  batched_latency < original_latency:
                for task in Queue:
                    if task.model == device.task.model:
                        device.batch_and_assign(task)
                        Queue.remove(task)
                        break

def naive_schedule(Devices, Queue, device, tail_latency):
    """Distribute the biggest task to the fastest GPU."""
    Queue.sort()
    if device.name == "GPU (2060)":
        task = Queue.pop(0)
        task.schedule_time = tail_latency
        device.assign(task)
    else:
        task = Queue.pop(len(Queue) - 1)
        task.schedule_time = tail_latency
        device.assign(task)


def evaluate(Devices, test_datas, scheduler, N=10, lazy=True):
    """Evaulate the scheduler with test data and repeats N times."""
    results = []
    lazy_results = []
    for j in tqdm(range(N)):
        test_data = test_datas[j]

        Tasks = deepcopy(test_data)
        tail_latency = emulate(Tasks, Devices, scheduler)
        results.append(tail_latency)

        if lazy:
            Tasks = deepcopy(test_data)
            tail_latency = emulate(Tasks, Devices, scheduler, lazy_batching_schedule)
            lazy_results.append(tail_latency)
        # results.append(sum([task.schedule_time - task.arrival_time for task in Tasks]))
    df = pd.DataFrame(results)
    mean = sum(results) / len(results)
    if lazy:
        df = pd.DataFrame(lazy_results)
        lazy_mean = sum(lazy_results) / len(lazy_results)
        return mean, lazy_mean
    return mean


def workload_generater(rate, size, Models):
    """Return a workload generated by poisson distribution."""
    test_data = []
    now = 0
    for _ in range(size):
        now += nextTime(rate)
        test_data.append(Task(random.choice(Models), arrival_time=now))

    return test_data


def best_schedule(Devices, Queue, device=None, tail_latency=None):
    # workload = 0
    # for task in Queue:
        # workload += task.size

    # for device in Devices:
        # if device.task:
            # workload += device.task.remain_time

    if device:
        if len(Queue):
            # assign
            assign_max = float('inf')
            assign_task = None
            for task in Queue:
                delta = task.model.gpu_size[device.GPUSpec.id] - task.size
                if assign_max > delta:
                    assign_max = delta
                    assign_task = task

            def do_assign():
                Queue.pop(Queue.index(assign_task))
                device.assign(assign_task)
            do_assign()
            
            # if assign_max >= move_and_assign_max:
                # decided = do_assign
            # else:
                # decided = do_move_and_assign
        else:
            pass
            # move
        # decided()
    # lazy batching
    else:
        lazy_batching_schedule(Devices, Queue)

def main():
    """Test the basic configuration and schedulers."""
    logging.basicConfig(filename='info.log',
                        encoding='utf-8',
                        level=logging.INFO)

    # Prepare the device and models.
    GPU_2060 = GPUSpec("RTX 2060", "2060")
    GPU_1080 = GPUSpec("GTX 1008 Ti", "1080")
    gpu_list = (GPU_2060, GPU_1080)

    Tasks = []
    Devices = []
    Models = []

    ResNet = get_resnet(gpu_list)
    # ResNet.adjust_latency(GPU_1080.id, 0.3)
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
    print("ResNet latency 2060:", sum(ResNet.layer_latency[1][GPU_2060.id]))
    print("ResNet latency 2060:", sum(ResNet.layer_latency[1][GPU_1080.id]))
    print(sum(ResNet.layer_latency[2][GPU_2060.id]))
    print(sum(ResNet.layer_latency[2][GPU_1080.id]))

    print(sum(AlexNet.layer_latency[1][GPU_2060.id]))
    print(sum(AlexNet.layer_latency[1][GPU_1080.id]))
    print(sum(AlexNet.layer_latency[2][GPU_2060.id]))
    print(sum(AlexNet.layer_latency[2][GPU_1080.id]))

    print(sum(RNN.layer_latency[1][GPU_2060.id]))
    print(sum(RNN.layer_latency[1][GPU_1080.id]))
    print(sum(RNN.layer_latency[2][GPU_2060.id]))
    print(sum(RNN.layer_latency[2][GPU_1080.id]))

    # exit()

    # Evaluate the scheduler with different poisson rate.
    for interval in (100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000,
                 10000000, 50000000, 100000000):
        rate = 1 / interval
        test_data = []

        N = 5
        tasks_size = 100
        test_datas = []
        for i in range(N):
            test_datas.append(workload_generater(rate, tasks_size, Models))

        naive_mean, lazy_naive_mean = evaluate(Devices, test_datas, naive_schedule, N)
        fifo_mean, lazy_fifo_mean = evaluate(Devices, test_datas, fifo_schedule, N)
        sjf_mean, lazy_sjf_mean = evaluate(Devices, test_datas, sjf_schedule, N)
        nonaive_mean, lazy_nonaive_mean = evaluate(Devices, test_datas, nonaive_schedule, N)
        best_mean, lazy_best_mean = evaluate(Devices, test_datas, best_schedule, N)
        batch_mean = evaluate(Devices, test_datas, batch_schedule, N, lazy=False)

        schedulers = ["Naive", "FIFO", "SJF", "Non-naive", "Best"]
        latencies = [naive_mean, fifo_mean, sjf_mean, nonaive_mean, best_mean]
        x = np.arange(len(schedulers))
        plt.cla()
        plt.bar(x, latencies)
        plt.xticks(x, schedulers)
        plt.xlabel('Schedulers')
        plt.ylabel('Latency')
        plt.savefig(f'{rate}.png')

        schedulers = ["Naive", "FIFO", "SJF", "Non-naive", "Best"]
        latencies = [lazy_naive_mean, lazy_fifo_mean, lazy_sjf_mean, lazy_nonaive_mean, lazy_best_mean]
        x = np.arange(len(schedulers))
        plt.bar(x, latencies)
        plt.xticks(x, schedulers)
        plt.xlabel('Schedulers')
        plt.ylabel('Latency')
        plt.savefig(f'lazy_{rate}.png')

        schedulers = ["Naive", "FIFO", "SJF", "Non-naive", "Best"]

        # Compare the speedup with naive scheduler and others.
        # print(f"{interval}:", naive_mean / fifo_mean, naive_mean / sjf_mean,
              # naive_mean / nonaive_mean, naive_mean/batch_mean)
        # print(f"{interval}:", naive_mean / fifo_mean, naive_mean / sjf_mean,
              # naive_mean / nonaive_mean)
        print(interval)
        print(naive_mean, lazy_naive_mean)
        print(fifo_mean, lazy_fifo_mean)
        print(sjf_mean, lazy_sjf_mean)
        print(nonaive_mean, lazy_nonaive_mean)
        print(best_mean, lazy_best_mean)
        print(batch_mean)
        print()


if __name__ == "__main__":
    main()
