import torch
import torch.nn as nn
from .ProgNet import ProgBlock
from .deform_conv2d import *

"""
A ProgBlock containing a single Conv2D layer (nn.Conv2d).
Activation function can be customized but defaults to nn.ReLU.
Stride, padding, dilation, groups, bias, and padding_mode can be set with layerArgs.
"""

class ProgConv2DBlock(ProgBlock):
    def __init__(self, inSize, outSize, kernelSize, numLaterals, activation = nn.ReLU(), layerArgs = dict()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.kernSize = kernelSize
        self.module = nn.Conv2d(inSize, outSize, kernelSize, **layerArgs)
        self.laterals = nn.ModuleList([nn.Conv2d(inSize, outSize, kernelSize, **layerArgs) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        return self.module(x)

    def runLateral(self, i, x):
        lat = self.laterals[i]
        return lat(x)

    def runActivation(self, x):
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "Conv2D"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["kernel_size"] = self.kernSize
        data["state_dict"] = self.module.state_dict()
        data["lateral_states"] = [l.state_dict() for l in self.laterals]
        return data

    def getShape(self):
        return (self.inSize, self.outSize)


"""
A ProgBlock containing a single Conv2D layer (nn.Conv2d) with Batch Normalization.
Activation function can be customized but defaults to nn.ReLU.
Stride, padding, dilation, groups, bias, and padding_mode can be set with layerArgs.
"""

class ProgConv2DBNBlock(ProgBlock):
    def __init__(self, inSize, outSize, kernelSize, numLaterals, activation = nn.ReLU(), layerArgs = dict(), bnArgs = dict()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.kernSize = kernelSize
        self.module = nn.Conv2d(inSize, outSize, kernelSize, **layerArgs)
        self.moduleBN = nn.BatchNorm2d(outSize, **bnArgs)
        self.laterals = nn.ModuleList([nn.Conv2d(inSize, outSize, kernelSize, **layerArgs) for _ in range(numLaterals)])
        self.lateralBNs = nn.ModuleList([nn.BatchNorm2d(outSize, **bnArgs) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        return self.moduleBN(self.module(x))

    def runLateral(self, i, x):
        lat = self.laterals[i]
        bn = self.lateralBNs[i]
        return bn(lat(x))

    def runActivation(self, x):
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "Conv2DBN"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["kernel_size"] = self.kernSize
        data["state_dict"] = self.module.state_dict()
        data["lateral_states"] = [l.state_dict() for l in self.laterals]
        return data

    def getShape(self):
        return (self.inSize, self.outSize)



"""
A ProgBlock containing a single ConvTranspose2D layer (nn.ConvTranspose2d) with Batch Normalization.
Activation function can be customized but defaults to nn.ReLU.
Stride, padding, dilation, groups, bias, and padding_mode can be set with layerArgs.
"""

class ProgConvTranspose2DBNBlock(ProgBlock):
    def __init__(self, inSize, outSize, kernelSize, numLaterals, activation = nn.ReLU(), layerArgs = dict(), bnArgs = dict()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.kernSize = kernelSize
        self.module = nn.ConvTranspose2d(inSize, outSize, kernelSize, **layerArgs)
        self.moduleBN = nn.BatchNorm2d(outSize, **bnArgs)
        self.laterals = nn.ModuleList([nn.ConvTranspose2d(inSize, outSize, kernelSize, **layerArgs) for _ in range(numLaterals)])
        self.lateralBNs = nn.ModuleList([nn.BatchNorm2d(outSize, **bnArgs) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        return self.moduleBN(self.module(x))

    def runLateral(self, i, x):
        lat = self.laterals[i]
        bn = self.lateralBNs[i]
        return bn(lat(x))

    def runActivation(self, x):
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "ProgConvTranspose2DBN"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["kernel_size"] = self.kernSize
        data["state_dict"] = self.module.state_dict()
        data["lateral_states"] = [l.state_dict() for l in self.laterals]
        return data

    def getShape(self):
        return (self.inSize, self.outSize)



class ProgDeformConv2DBlock(ProgBlock):
    def __init__(self, inSize, outSize, kernelSize, numLaterals, activation = nn.ReLU(), layerArgs = dict()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.kernSize = kernelSize
        self.module = DeformConv2d(inSize, outSize, kernelSize, **layerArgs, modulation=True)
        self.laterals = nn.ModuleList([DeformConv2d(inSize, outSize, kernelSize, **layerArgs, modulation=True) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        return self.module(x)

    def runLateral(self, i, x):
        lat = self.laterals[i]
        return lat(x)

    def runActivation(self, x):
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "DeformConv2D"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["kernel_size"] = self.kernSize
        data["state_dict"] = self.module.state_dict()
        data["lateral_states"] = [l.state_dict() for l in self.laterals]
        return data

    def getShape(self):
        return (self.inSize, self.outSize)


class ProgDeformConv2DBNBlock(ProgBlock):
    def __init__(self, inSize, outSize, kernelSize, numLaterals, activation = nn.ReLU(), layerArgs = dict(), bnArgs = dict()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.kernSize = kernelSize
        self.module = DeformConv2d(inSize, outSize, kernelSize, **layerArgs, modulation=True)
        self.moduleBN = nn.BatchNorm2d(outSize, **bnArgs)
        self.laterals = nn.ModuleList([DeformConv2d(inSize, outSize, kernelSize, **layerArgs, modulation=True) for _ in range(numLaterals)])
        self.lateralBNs = nn.ModuleList([nn.BatchNorm2d(outSize, **bnArgs) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        return self.moduleBN(self.module(x))

    def runLateral(self, i, x):
        lat = self.laterals[i]
        bn = self.lateralBNs[i]
        return bn(lat(x))

    def runActivation(self, x):
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "DeformConv2DBN"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["kernel_size"] = self.kernSize
        data["state_dict"] = self.module.state_dict()
        data["lateral_states"] = [l.state_dict() for l in self.laterals]
        return data

    def getShape(self):
        return (self.inSize, self.outSize)
