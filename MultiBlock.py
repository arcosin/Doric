
import torch.nn as nn
from .ProgNet import ProgBlock, ProgInertBlock


class PassBlock(ProgInertBlock):
    def __init__(self):
        super().__init__()

    def runBlock(self, x):
        return x

    def runActivation(self, x):
        return x

    def getData(self):
        data = dict()
        data["type"] = "Pass"
        return data




class MultiBlock(ProgBlock):
    def __init__(self, subBlocks):
        super().__init__()
        self.channels = nn.ModuleList(subBlocks)

    def runBlock(self, x):
        outs = []
        self._checkInput(x)
        for i, inp in enumerate(x):
            b = self.channels[i]
            outs.append(b.runBlock(inp))
        return outs

    def runLateral(self, j, x):
        outs = []
        self._checkInput(x)
        for i, inp in enumerate(x):
            b = self.channels[i]
            if b.isLateralized():
                outs.append(b.runLateral(j, inp))
            else:
                outs.append(None)
        return outs

    def runActivation(self, x):
        outs = []
        self._checkInput(x)
        for i, inp in enumerate(x):
            b = self.channels[i]
            a = b.runActivation(inp)
            outs.append(a)
        return outs

    def getData(self):
        data = dict()
        data["type"] = "Multi"
        data["subblocks"] = [sb.getData() for sb in self.channels]
        return data

    def _checkInput(self, x):
        if len(x) != len(self.channels):
            errStr = "[Doric]: Input must be a python iterable with size equal to the number of channels in the MultiBlock (%d)." % len(self.channels)
            raise ValueError(errStr)


#===============================================================================
