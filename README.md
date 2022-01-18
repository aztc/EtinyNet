# EtinyNet

EtinyNet is an extremely tiny CNN backbone for Tiny Machine Learning (TinyML) that aims at executing AI workloads on low-power & low-cost IoT devices with limitted memory, such as Microcontrollor (MCU) and compact Filed Programable Gata Array (FPGA). 

We currently provide two settings of EtinyNet-1.0 and EtinyNet-0.75, which have only 477K and parameters and 360K parameters. The performance of these two models on ImageNet and comparisons with other state-of-the-art lightweight models are shwon below.


We deploy the int8 quantized EtinyNet-1.0 on STM32H743 MCU for running object classification and detection. Results are in Tabel 2.



In fact, the EtinyNet can exhibit its powerful performance on the specially designed CNN accelerator TinyNPU. Since EtinyNet comsumes only ~800KB memory (except fully-connected layer), TinyNPU can runs it in a single-chip mannar without accessing off-chip memory, saving much energy and latency caused by data transmission. We build a system based on TinyNPU + MCU(STM32L4R9), in which the MCU runs pre-processsing and post-processing while TinyNPU runs CNN workloads. TinyNPU stores weights and feature maps on chip and connects with MCU via SDIO/SPI interface to transmite images and results. The system has a really simple working pipeline as: 1) MCU sends image to TinyNPU, 2) TinyNPU runs CNNs, 3) TinyNPU sends results back. 

Here's a video presents the prototype system.

[![NetFlix on UWP](https://i9.ytimg.com/vi/mIZPxtJ-9EY/mq3.jpg?sqp=CPi5m48G&rs=AOn4CLDUoZkhxVe61lq4CFTQF-2xTauSSg)](https://www.youtube.com/watch?v=mIZPxtJ-9EY)


