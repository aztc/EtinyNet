from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from gluoncv.nn import ReLU6
import mxnet as mx
from gluoncv.model_zoo.ssd import get_ssd
from gluoncv.model_zoo import get_model
from gluoncv.nn.coder import MultiPerClassDecoder, NormalizedBoxCenterDecoder
from gluoncv.model_zoo.ssd.anchor import SSDAnchorGenerator
from mxnet import autograd
import numpy as np


class Clip(HybridBlock):
    def __init__(self, **kwargs):
        super(Clip, self).__init__(**kwargs)
    def hybrid_forward(self, F, x):
        return F.clip(x, -12, 12, name="clip")
    

# pylint: disable= too-many-arguments
def _add_conv(out, channels=1, kernel=1, stride=1, pad=0, norm=True, in_channels=0, quantized=True,
              num_group=1, active=True, relu6=False, norm_layer=BatchNorm, norm_kwargs=None):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False,in_channels=in_channels))
    if active:
        out.add(ReLU6() if relu6 else nn.Activation('relu'))
    if norm:
        out.add(norm_layer(scale=True, center=True, **({} if norm_kwargs is None else norm_kwargs)))


class LinearBottleneck(nn.HybridBlock):
    r"""LinearBottleneck used in MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    t : int
        Layer expansion ratio.
    stride : int
        stride
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, channels, stride, shortcut,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut1 = (stride == 1 and channels[0] == channels[1] and shortcut)
        self.use_shortcut2 = shortcut
        with self.name_scope():
            self.out1 = nn.HybridSequential() # 1x1
            self.out2 = nn.HybridSequential() # 1x1
            
            _add_conv(self.out1,
                      in_channels=channels[0],
                      channels=channels[0],
                      kernel=3,
                      stride=stride,
                      pad=1,
                      num_group=channels[0],
                      active=False,
                      relu6=False,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            _add_conv(self.out1,
                      in_channels=channels[0],
                      channels=channels[1],
                      active=True,
                      relu6=False,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            _add_conv(self.out2,
                      in_channels=channels[1],
                      channels=channels[1], 
                      kernel=3, 
                      stride=1, 
                      pad=1,
                      num_group=channels[1],
                      active=True, 
                      relu6=False,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                        
            
    def hybrid_forward(self, F, x):    
        out = self.out1(x)
        if self.use_shortcut1:        
            out = F.elemwise_add(out, x)
        x = out
        out = self.out2(out)
        if self.use_shortcut2:
            out = F.elemwise_add(out, x)   
        return out



class Etinynet(nn.HybridBlock):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.
    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controlling the model size. The actual number of channels
        is equal to the original channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, multiplier=1.0, classes=1000, norm_layer=BatchNorm, norm_kwargs=None, 
                  ctx=cpu(), root='', pretrained=False, **kwargs):
        super(Etinynet, self).__init__(**kwargs)
        
        with self.name_scope():
            self.features1 = nn.HybridSequential(prefix='features1_')
            self.features2 = nn.HybridSequential(prefix='features2_')
            self.features3 = nn.HybridSequential(prefix='features2_')
            with self.features1.name_scope():
                _add_conv(self.features1, int(32 * multiplier), kernel=3,
                          stride=(2,2), pad=1, relu6=False,in_channels=3, quantized=True,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                
                self.features1.add(nn.MaxPool2D((2,2)))
                channels_group = [[32, 32],  [32, 32], [32, 32], [32, 32],
                                  [32, 128], [128, 128], [128, 128], [128, 128]]
                strides =   [1,1,1,1] + [2,1,1,1] 
                shortcuts = [0,0,0,0] + [0,0,0,0]
                for cg, s, sc in zip(channels_group, strides, shortcuts):
                    self.features1.add(LinearBottleneck(channels=np.int32(np.array(cg)*multiplier),
                                                       stride=s,
                                                       shortcut=sc,
                                                       norm_layer=norm_layer,
                                                       norm_kwargs=norm_kwargs))
                
                           
                channels_group = [[128, 192], [192, 192], [192, 192]]
                strides =   [2,1,1]
                shortcuts = [1,1,1]
                for cg, s, sc in zip(channels_group, strides, shortcuts):
                    self.features2.add(LinearBottleneck(channels=np.int32(np.array(cg)*multiplier),
                                                       stride=s,
                                                       shortcut=sc,
                                                       norm_layer=norm_layer,
                                                       norm_kwargs=norm_kwargs))
                
                
                channels_group = [[192, 256], [256, 256], [256, 512]]
                strides =   [2,1,1]
                shortcuts = [1,1,1]
                for cg, s, sc in zip(channels_group, strides, shortcuts):
                    self.features3.add(LinearBottleneck(channels=np.int32(np.array(cg)*multiplier),
                                                       stride=s,
                                                       shortcut=sc,
                                                       norm_layer=norm_layer,
                                                       norm_kwargs=norm_kwargs))
                self.avg = nn.GlobalAvgPool2D()


            self.output = nn.HybridSequential(prefix='output_')
            with self.output.name_scope():
                 self.output.add(
                     nn.Conv2D(classes, 1, in_channels=int(512*multiplier),prefix='pred_'),
                     nn.Flatten())
          
        
        
    def hybrid_forward(self, F, x):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)  
        x = self.avg(x3)
        x = self.output(x)
        return x



if __name__ == "__main__": 
    model = Etinynet()
    model.initialize()
    model.summary(mx.ndarray.zeros((1,3,256,256)))
        

