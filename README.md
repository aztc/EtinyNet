# EtinyNet

EtinyNet is an extremely tiny CNN backbone for Tiny Machine Learning (TinyML) that aims at executing AI workloads on low-power & low-cost IoT devices with limitted memory, such as Microcontrollor (MCU) , compact Filed Programable Gata Array (FPGA) and small footprint CNNs accelerator. 

We currently provide two settings of EtinyNet-1.0 and EtinyNet-0.75, which have only 477K and parameters and 360K parameters. The performance of these two models on ImageNet and comparisons with other state-of-the-art lightweight models are shwon below.

Table 1. Comparison of state-of-the-art small networks over classification accuracy, the model size and MAdds on ImageNet-1000 dataset. “-” mean no reported results available. The input size is 224x224.
| Model| Params.(M) |  Top-1 Acc. (%)| Top-5 Acc. (%)|
| ---- | -- |-- |-- |
| MobileNeXt-0.35 | 1.8 |64.7|-|
| MnasNet-A1-0.35 | 1.7 |64.4|85.1|
| MobileNetV2-0.35 | 1.7 |60.3|82.9
| MicroNet-M3 | 1.6 |61.3|82.9|
| MobileNetV3-Small-0.35 | 1.6 |58.0|-|
| ShuffleNetV2-0.5 | 1.4 |61.1|82.6|
| MicroNet-M2 | 1.4 |58.2|80.1|
| MobileNetV2-0.15 | 1.4 |55.1|-|
| MobileNetV1-0.5 | 1.3 |61.7|83.6|
| EfficientNet-B | 1.3 |56.7|79.8|
| **EtinyNet-1.0** | **0.98** |**65.5**|**86.2**|
| **EtinyNet-0.75** | **0.68** |**62.2**|**84.0**|



We deploy the int8 quantized EtinyNet-1.0 on STM32H743 MCU for running object classification and detection. Results are in Tabel 2.

Tabel 2. Comparison to MCU designs on ImageNet. EtinyNet obtains the record accuracy of64.7% and 65.8% on STM32F412 and STM32F746.
| Model| STM32F412 |  STM32F746 |
| ---- | -- |-- |
| Rusciet al.  | 60.2% |--|
| MCUNet       | 62.2% |63.5%|
| EtinyNet-1.0 | **64.7%** |**65.8%**|

In fact, the EtinyNet can exhibit its powerful performance on the specially designed CNN accelerator Neural Co-processor (NCP). Since EtinyNet comsumes only ~800KB memory (except fully-connected layer), NCP can run it in a single-chip mannar without accessing off-chip memory, saving much energy and latency caused by data transmission. We build a system based on NCP + MCU(STM32L4R9), in which the MCU runs pre-processsing and post-processing while NCP runs EtinyNet. NCP stores weights and feature maps on chip and connects with MCU via SDIO/SPI interface to transmite images and results. The system has a really simple working pipeline as: 1) MCU sends image to NCP, 2) NCP runs EtinyNet, 3) TinyNPU sends results back. With the NCP, we prompt the throughput of entire system to 30fps and reache an extremelly low processing power of 160mW (MCU + NCP)。

Here's a video that presents the prototype system.

[![EtinyNet](https://i9.ytimg.com/vi/mIZPxtJ-9EY/mq3.jpg?sqp=COju4ZAG&rs=AOn4CLDglN9ujGc3h1syZAd-s9PNYzD9-Q)](https://www.youtube.com/watch?v=mIZPxtJ-9EY)


We provide the training code for EtinyNet-1.0 (no quantization) as well as the coresponding test code and well-trained parameters, as listed below:

1) train_imagenet.py: training code. The default input size is 224, 300 epoches and  128 batchsize x 8 gpus.

2) etinynet.py: EtinyNet-1.0.

3) 0.6553-imagenet-mobilenet_lite313_477k_nownorm_4433_224-293-best.py: well-trained parameters for EtinyNet-1.0 (no quantization)

4) test_imagenet.py : training code.

The MXNet toolbox and '.rec' compressed ImageNet data file were used for training efficiency. Please refer to https://mxnet.incubator.apache.org/versions/1.9.0/ for more detail about '.rec' data.


