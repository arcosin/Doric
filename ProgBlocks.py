
import torch
import torch.nn as nn
from .ProgNet import ProgBlock


I_FUNCTION = (lambda x : x)


#=================================<ProgBlocks>=================================#

"""
A ProgBlock containing a single fully connected layer (nn.Linear).
Activation function can be customized but defaults to nn.ReLU.
"""
class ProgDenseBlock(ProgBlock):
    def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU(), skipConn = False, lambdaSkip = I_FUNCTION):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.skipConn = skipConn
        self.skipVar = None
        self.skipFunction = lambdaSkip
        self.module = nn.Linear(inSize, outSize)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        if self.skipConn:
            self.skipVar = x
        return self.module(x)

    def runLateral(self, i, x):
        lat = self.laterals[i]
        return lat(x)

    def runActivation(self, x):
        if self.skipConn and self.skipVar is not None:
            x = x + self.skipFunction(self.skipVar)
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "Dense"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["skip"] = self.skipConn
        return data

    def getShape(self):
        return (self.inSize, self.outSize)




"""
A ProgBlock containing a single fully connected layer (nn.Linear) and a batch norm.
Activation function can be customized but defaults to nn.ReLU.
"""
class ProgDenseBNBlock(ProgBlock):
    def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU(), bnArgs = dict(), skipConn = False, lambdaSkip = I_FUNCTION):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.skipConn = skipConn
        self.skipVar = None
        self.skipFunction = lambdaSkip
        self.module = nn.Linear(inSize, outSize)
        self.moduleBN = nn.BatchNorm1d(outSize, **bnArgs)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        self.lateralBNs = nn.ModuleList([nn.BatchNorm1d(outSize, **bnArgs) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        if self.skipConn:
            self.skipVar = x
        return self.moduleBN(self.module(x))

    def runLateral(self, i, x):
        lat = self.laterals[i]
        bn = self.lateralBNs[i]
        return bn(lat(x))

    def runActivation(self, x):
        if self.skipConn and self.skipVar is not None:
            x = x + self.skipFunction(self.skipVar)
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "DenseBN"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["skip"] = self.skipConn
        return data

    def getShape(self):
        return (self.inSize, self.outSize)



'''

"""
A ProgBlock containing a single Conv2D layer (nn.Conv2d).
Activation function can be customized but defaults to nn.ReLU.
Stride, padding, dilation, groups, bias, and padding_mode can be set with layerArgs.
"""

class ProgConv2DBlock(ProgBlock):
    def __init__(self, inSize, outSize, kernelSize, numLaterals, activation = nn.ReLU(), layerArgs = dict(), skipConn = False, lambdaSkip = I_FUNCTION):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.skipConn = skipConn
        self.skipVar = None
        self.skipFunction = lambdaSkip
        self.kernSize = kernelSize
        self.module = nn.Conv2d(inSize, outSize, kernelSize, **layerArgs)
        self.laterals = nn.ModuleList([nn.Conv2d(inSize, outSize, kernelSize, **layerArgs) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        if self.skipConn:
            self.skipVar = x
        return self.module(x)

    def runLateral(self, i, x):
        lat = self.laterals[i]
        return lat(x)

    def runActivation(self, x):
        if self.skipConn and self.skipVar is not None:
            x = x + self.skipFunction(self.skipVar)
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "Conv2D"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["kernel_size"] = self.kernSize
        data["skip"] = self.skipConn
        return data





"""
A ProgBlock containing a single Conv2D layer (nn.Conv2d) with Batch Normalization.
Activation function can be customized but defaults to nn.ReLU.
Stride, padding, dilation, groups, bias, and padding_mode can be set with layerArgs.
"""

class ProgConv2DBNBlock(ProgBlock):
    def __init__(self, inSize, outSize, kernelSize, numLaterals, activation = nn.ReLU(), layerArgs = dict(), bnArgs = dict(), skipConn = False, lambdaSkip = I_FUNCTION):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.skipConn = skipConn
        self.skipVar = None
        self.skipFunction = lambdaSkip
        self.kernSize = kernelSize
        self.module = nn.Conv2d(inSize, outSize, kernelSize, **layerArgs)
        self.moduleBN = nn.BatchNorm2d(outSize, **bnArgs)
        self.laterals = nn.ModuleList([nn.Conv2d(inSize, outSize, kernelSize, **layerArgs) for _ in range(numLaterals)])
        self.lateralBNs = nn.ModuleList([nn.BatchNorm2d(outSize, **bnArgs) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        if self.skipConn:
            self.skipVar = x
        return self.moduleBN(self.module(x))

    def runLateral(self, i, x):
        lat = self.laterals[i]
        bn = self.lateralBNs[i]
        return bn(lat(x))

    def runActivation(self, x):
        if self.skipConn and self.skipVar is not None:
            x = x + self.skipFunction(self.skipVar)
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "Conv2DBN"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["kernel_size"] = self.kernSize
        data["skip"] = self.skipConn
        return data


"""
A ProgBlock containing a single ConvTranspose2D layer (nn.ConvTranspose2d) with Batch Normalization.
Activation function can be customized but defaults to nn.ReLU.
Stride, padding, dilation, groups, bias, and padding_mode can be set with layerArgs.
"""

class ProgConvTranspose2DBNBlock(ProgBlock):
    def __init__(self, inSize, outSize, kernelSize, numLaterals, activation = nn.ReLU(), layerArgs = dict(), bnArgs = dict(), skipConn = False, lambdaSkip = I_FUNCTION):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.skipConn = skipConn
        self.skipVar = None
        self.skipFunction = lambdaSkip
        self.kernSize = kernelSize
        self.module = nn.ConvTranspose2d(inSize, outSize, kernelSize, **layerArgs)
        self.moduleBN = nn.BatchNorm2d(outSize, **bnArgs)
        self.laterals = nn.ModuleList([nn.ConvTranspose2d(inSize, outSize, kernelSize, **layerArgs) for _ in range(numLaterals)])
        self.lateralBNs = nn.ModuleList([nn.BatchNorm2d(outSize, **bnArgs) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        if self.skipConn:
            self.skipVar = x
        return self.moduleBN(self.module(x))

    def runLateral(self, i, x):
        lat = self.laterals[i]
        bn = self.lateralBNs[i]
        return bn(lat(x))

    def runActivation(self, x):
        if self.skipConn and self.skipVar is not None:
            x = x + self.skipFunction(self.skipVar)
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "ProgConvTranspose2DBN"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["kernel_size"] = self.kernSize
        data["skip"] = self.skipConn
        return data

    def getShape(self):
        return (self.inSize, self.outSize)

'''


#===============================================================================
