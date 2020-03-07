
import torch
import torch.nn as nn
from .ProgNet import ProgBlock, ProgMultiBlock


#=================================<ProgBlocks>=================================#

"""
A ProgBlock containing a single fully connected layer (nn.Linear).
Activation function can be customized but defaults to nn.ReLU.
"""
class ProgDenseBlock(ProgBlock):
    def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module = nn.Linear(inSize, outSize)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
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
        data["type"] = "Dense"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["state_dict"] = self.module.state_dict()
        data["lateral_states"] = [l.state_dict() for l in self.laterals]
        return data

    def getShape(self):
        return (self.inSize, self.outSize)






"""
An inert ProgBlock that simply runs the input through python lambda functions.
Good for resizing or other non-learning ops.
"""
class ProgLambdaBlock(ProgInertBlock):
    def __init__(self, inSize, outSize, lambdaMod):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module = lambdaMod
        self.activation = (lambda x: x)

    def runBlock(self, x):
        return self.module(x)

    def runLateral(self, i, x):
        raise NotImplementedError

    def runActivation(self, x):
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "Lambda"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        return data

    def getShape(self):
        return (self.inSize, self.outSize)






"""
A ProgBlock containing a single Conv2D layer (nn.Conv2d).
Activation function can be customized but defaults to nn.ReLU.
Stride, padding, dilation, groups, bias, and padding_mode can be set with layerArgs.
"""
'''
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
'''






"""
A ProgBlock containing a single fully connected layer (nn.Linear) and a batch norm.
Activation function can be customized but defaults to nn.ReLU.
"""
class ProgDenseBNBlock(ProgBlock):
    def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU(), bnArgs = dict()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module = nn.Linear(inSize, outSize)
        self.moduleBN = nn.BatchNorm1d(outSize, **bnArgs)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        self.lateralBNs = nn.ModuleList([nn.BatchNorm1d(outSize, **bnArgs) for _ in range(numLaterals)])
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
        data["type"] = "DenseBN"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["state_dict"] = self.module.state_dict()
        data["lateral_states"] = [l.state_dict() for l in self.laterals]
        return data

    def getShape(self):
        return (self.inSize, self.outSize)






"""
A MultiBlock with dense layers and passes.
passList is a list of booleans that are true when the input should be passed through untouched.
"""
class ProgMultiDense(ProgMultiBlock):
    def __init__(self, inSizes, outSizes, numLaterals, passList, activation = nn.ReLU()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSizes = [x for x in inSizes]
        self.outSizes = [x for x in outSizes]
        self.module = nn.ModuleList()
        self.laterals = nn.ModuleList()
        for i, p in enumerate(passList):
            if not p:
                self.module.append(nn.Linear(inSizes[i], outSizes[i]))
                lat = nn.ModuleList([nn.Linear(inSizes[i], outSizes[i]) for _ in range(numLaterals)])
                self.laterals.append(lat)
            else:
                self.module.append(None)
                self.laterals.append(None)
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation
        self.passDescriptor = [x for x in passList]

    def runBlock(self, x):
        outs = []
        errStr = "Input list for runBlock must be the same size as the pass list."
        assert len(x) == len(self.passDescriptor), errStr
        for i, input in enumerate(x):
            modu = self.module[i]
            if modu is not None:   outs.append(modu(input))
            else:                  outs.append(input)
        return outs

    def runLateral(self, i, x):
        outs = []
        errStr = "Input list for runLateral must be the same size as the pass list."
        assert len(x) == len(self.passDescriptor), errStr
        for j, input in enumerate(x):
            lats = self.laterals[j]
            if lats is not None:
                lat = lats[i]
                outs.append(lat(input))
            else:
                outs.append(input)
        return outs

    def runActivation(self, x):
        outs = []
        errStr = "Input list for runActivation must be the same size as the pass list."
        assert len(x) == len(self.passDescriptor), errStr
        for i, input in enumerate(x):
            modu = self.module[i]
            if modu is not None:
                outs.append(self.activation(input))
            else:
                outs.append(input)
        return outs

    def getPassDescriptor(self):
        return [x for x in self.passDescriptor]

    def getData(self):
        data = dict()
        data["type"] = "Multi-Dense"
        data["pass_descriptor"] = self.passDescriptor
        data["input_sizes"] = self.inSizes
        data["output_sizes"] = self.outSizes
        modList = []
        for m in self.module:
            if m is not None:   modList.append(m.state_dict())
            else:               modList.append(None)
        data["state_dicts"] = modList
        latLists = []
        for lat in self.laterals:
            if lat is not None:   latList = [l.state_dict() for l in lat]
            else:                 latList.append(None)
        data["lateral_states"] = latLists
        return data

    def getShape(self):
        return (self.inSizes, self.outSizes)






"""
A MultiBlock with dense layers and passes.
passList is a list of booleans that are true when the input should be passed through untouched.
"""
class ProgMultiDenseBN(ProgMultiBlock):
    def __init__(self, inSizes, outSizes, numLaterals, passList, activation = nn.ReLU()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSizes = [x for x in inSizes]
        self.outSizes = [x for x in outSizes]
        self.module = nn.ModuleList()
        self.moduleBNs = nn.ModuleList()
        self.laterals = nn.ModuleList()
        self.lateralBNs = nn.ModuleList()
        for i, p in enumerate(passList):
            if not p:
                self.module.append(nn.Linear(inSizes[i], outSizes[i]))
                self.moduleBNs.append(nn.BatchNorm1d(outSizes[i]))
                lat = nn.ModuleList([nn.Linear(inSizes[i], outSizes[i]) for _ in range(numLaterals)])
                self.laterals.append(lat)
                latBN = nn.ModuleList([nn.BatchNorm1d(outSizes[i]) for _ in range(numLaterals)])
                self.lateralBNs.append(latBN)
            else:
                self.module.append(None)
                self.moduleBNs.append(None)
                self.laterals.append(None)
                self.lateralBNs.append(None)
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation
        self.passDescriptor = [x for x in passList]

    def runBlock(self, x):
        outs = []
        errStr = "Input list for runBlock must be the same size as the pass list."
        assert len(x) == len(self.passDescriptor), errStr
        for i, input in enumerate(x):
            modu = self.module[i]
            bn = self.moduleBNs[i]
            if modu is not None:   outs.append(bn(modu(input)))
            else:                  outs.append(input)
        return outs

    def runLateral(self, i, x):
        outs = []
        errStr = "Input list for runLateral must be the same size as the pass list."
        assert len(x) == len(self.passDescriptor), errStr
        for j, input in enumerate(x):
            lats = self.laterals[j]
            bns = self.lateralBNs[j]
            if lats is not None:
                lat = lats[i]
                bn = bns[i]
                outs.append(bn(lat(input)))
            else:
                outs.append(input)
        return outs

    def runActivation(self, x):
        outs = []
        errStr = "Input list for runActivation must be the same size as the pass list."
        assert len(x) == len(self.passDescriptor), errStr
        for i, input in enumerate(x):
            modu = self.module[i]
            if modu is not None:
                outs.append(self.activation(input))
            else:
                outs.append(input)
        return outs

    def getPassDescriptor(self):
        return [x for x in self.passDescriptor]

    def getData(self):
        data = dict()
        data["type"] = "Multi-Dense"
        data["pass_descriptor"] = self.passDescriptor
        data["input_sizes"] = self.inSizes
        data["output_sizes"] = self.outSizes
        modList = []
        for m in self.module:
            if m is not None:   modList.append(m.state_dict())
            else:               modList.append(None)
        data["state_dicts"] = modList
        latLists = []
        for lat in self.laterals:
            if lat is not None:   latList = [l.state_dict() for l in lat]
            else:                 latList.append(None)
        data["lateral_states"] = latLists
        return data

    def getShape(self):
        return (self.inSizes, self.outSizes)






"""
A modified ProgMultiDense that sums all outputs and passed inputs to produce a single output.
The single output can be used as input to a simple ProgBlock.
"""
class ProgMultiDenseSum(ProgMultiDense):
    def __init__(self, inSizes, outSizes, numLaterals, passList, activation = nn.ReLU()):
        super().__init__(inSizes, outSizes, numLaterals, passList, activation = activation)
        errStr = "All outputs must be the same size for a dense-sum block."
        assert all(x == outSizes[0] for x in outSizes), errStr

    def runActivation(self, x):
        out = None
        errStr = "Input list for runActivation must be the same size as the pass list."
        assert len(x) == len(self.passDescriptor), errStr
        for inp in x:
            if out is None:   out = inp
            else:             out = torch.add(out, inp)
        return self.activation(out)

    def getData(self):
        data = dict()
        data["type"] = "Multi-Dense-Sum"
        data["pass_descriptor"] = self.passDescriptor
        data["input_sizes"] = self.inSizes
        data["output_sizes"] = self.outSizes
        modList = []
        for m in self.module:
            if m is not None:   modList.append(m.state_dict())
            else:               modList.append(None)
        data["state_dicts"] = modList
        latLists = []
        for lat in self.laterals:
            if lat is not None:   latList = [l.state_dict() for l in lat]
            else:                 latList.append(None)
        data["lateral_states"] = latLists
        return data






"""
A modified ProgMultiDense that concatenates the outputs and passed inputs to produce a single output.
The single output can be used as input to a simple ProgBlock.
"""
class ProgMultiDenseConcat(ProgMultiDense):
    def __init__(self, inSizes, outSizes, numLaterals, passList, activation = nn.ReLU()):
        super().__init__(inSizes, outSizes, numLaterals, passList, activation = activation)

    def runActivation(self, x):
        out = None
        errStr = "Input list for runActivation must be the same size as the pass list."
        assert len(x) == len(self.passDescriptor), errStr
        for input in x:
            if out is None:   out = input
            else:             out = torch.cat([out, input])
        return self.activation(out)

    def getData(self):
        data = dict()
        data["type"] = "Multi-Dense-Concat"
        data["pass_descriptor"] = self.passDescriptor
        data["input_sizes"] = self.inSizes
        data["output_sizes"] = self.outSizes
        modList = []
        for m in self.module:
            if m is not None:   modList.append(m.state_dict())
            else:               modList.append(None)
        data["state_dicts"] = modList
        latLists = []
        for lat in self.laterals:
            if lat is not None:   latList = [l.state_dict() for l in lat]
            else:                 latList.append(None)
        data["lateral_states"] = latLists
        return data


#===============================================================================
