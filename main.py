from objects import DeviceType, GPUSpec, Device, Model, Task
from models.resnet import get_resnet
from models.alexnet import get_alexnet
from models.rnn import get_rnn



def emulate(Tasks, Devices):
    tail_latency = 0
    while True:
        min_delay = float("inf")
        for device in Devices:
            if device.type == DeviceType.GPU:
                if device.task:
                    min_delay = min((min_delay, device.remain_time))

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
                    else:
                        pass

        if min_delay == float("inf"):
            if len(Tasks) == 0:
                break
            else:
                for device in Devices:
                    if device.type == DeviceType.GPU:
                        if not device.task and len(Tasks):
                            schedule(Tasks, device)
        else:
            tail_latency += min_delay
    print(f"[Result] Tail latency: {tail_latency}")

def schedule(Tasks, device):
    device.assign(Tasks.pop(0))


def main():
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

    Tasks = [Task(ResNet), Task(ResNet), Task(AlexNet), Task(AlexNet)]
    emulate(Tasks, Devices)


if __name__ == "__main__":
    main()
