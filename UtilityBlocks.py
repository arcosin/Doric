
import torch

from .ProgNet import ProgInertBlock


IO_STATE_IN = 0
IO_STATE_OUT = 1

I_FUNCTION = (lambda x : x)




"""
    An inert ProgBlock that simply runs the input through python lambda functions.
    Good for resizing or other non-learning ops.
"""
class ProgLambda(ProgInertBlock):
    def __init__(self, lambdaMod):
        super().__init__()
        self.module = lambdaMod

    def runBlock(self, x):
        return self.module(x)

    def runActivation(self, x):
        return x

    def getData(self):
        data = dict()
        data["type"] = "Lambda"
        return data



"""
    A convenience reshaping inert ProgBlock.
"""
class ProgReshape(ProgInertBlock):
    def __init__(self, shape):
        super().__init__()
        self.sh = shape

    def runBlock(self, x):
        return torch.reshape(x, self.sh)

    def runActivation(self, x):
        return x

    def getData(self):
        data = dict()
        data["type"] = "Reshape"
        data["new_shape"] = str(self.sh)
        return data



class ProgSkip(ProgInertBlock):
    def __init__(self, lambdaActivation = I_FUNCTION, lambdaSkip = I_FUNCTION):
        super().__init__()
        self.skip = None
        self.ioState = IO_STATE_IN
        self.activation = lambdaActivation
        self.skipFunction = lambdaSkip

    def runBlock(self, x):
        if self.ioState == IO_STATE_IN:
            self.skip = x
            self.ioState = IO_STATE_OUT
            return x
        else:
            ret = self.skip
            self.skip = None
            self.ioState = IO_STATE_IN
            return x + self.skipFunction(ret)

    def runActivation(self, x):
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "Skip"
        data["id"] = str(id(self))
        return data





#===============================================================================
