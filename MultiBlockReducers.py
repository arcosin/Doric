
import torch
import torch.nn as nn
from .ProgNet import ProgInertBlock


class SumReducer(ProgInertBlock):
    def __init__(self, numChannels, sumIndicator = None):
        super().__init__()
        self.numChannels = numChannels
        if sumIndicator is None:
            self.indicator = [True] * numChannels
        else:
            self.indicator = sumIndicator

    def runBlock(self, x):
        outs = []
        self._checkInput(x)
        summedChannel = self.indicator.index(True)
        currSum = x[summedChannel]
        for i, inp in enumerate(x):
            if i == summedChannel:
                outs.append(None)
            elif self.indicator[i]:
                currSum = torch.add(currSum, inp)
            else:
                outs.append(inp)
        outs[summedChannel] = currSum
        if len(outs) == 1:
            outs = outs[0]
        return outs

    def runActivation(self, x):
        return x

    def getData(self):
        data = dict()
        data["type"] = "Sum_Reducer"
        return data

    def _checkInput(self, x):
        if len(x) != self.numChannels:
            errStr = "[Doric]: Input must be a python iterable with size equal to the number of channels in the Reducer (%d)." % self.numChannels
            raise ValueError(errStr)






class ConcatReducer(ProgInertBlock):
    def __init__(self, numChannels, catIndicator = None):
        super().__init__()
        self.numChannels = numChannels
        if sumIndicator is None:
            self.indicator = [True] * numChannels
        else:
            self.indicator = catIndicator

    def runBlock(self, x):
        outs = []
        self._checkInput(x)
        cattedChannel = self.indicator.index(True)
        currCat = x[cattedChannel]
        for i, inp in enumerate(x):
            if i == cattedChannel:
                outs.append(None)
            elif self.indicator[i]:
                currCat = torch.cat([currCat, inp])
            else:
                outs.append(inp)
        outs[cattedChannel] = currCat
        return outs

    def runActivation(self, x):
        return x

    def getData(self):
        data = dict()
        data["type"] = "Concat_Reducer"
        return data

    def _checkInput(self, x):
        if len(x) != self.numChannels:
            errStr = "[Doric]: Input must be a python iterable with size equal to the number of channels in the Reducer (%d)." % self.numChannels
            raise ValueError(errStr)


#===============================================================================
