

import torch
import torch.nn as nn






#==============================<Abstract Classes>==============================#

"""
Class that acts as the base building-blocks of ProgNets.
Includes a module (usually a single layer),
a set of lateral modules, and an activation.
"""
class ProgBlock(nn.Module):
    """
    Runs the block on input x.
    Returns output tensor or list of output tensors.
    """
    def runBlock(self, x):
        raise NotImplementedError

    """
    Runs lateral i on input x.
    Returns output tensor or list of output tensors.
    """
    def runLateral(self, i, x):
        raise NotImplementedError

    """
    Runs activation of the block on x.
    Returns output tensor or list of output tensors.
    """
    def runActivation(self, x):
        raise NotImplementedError

    """
    Returns a dictionary of data about the block.
    """
    def getData(self):
        raise NotImplementedError

    """
    Returns True if block is meant to contain laterals.
    Returns False if block is meant to be a utility with not lateral inputs.
    Default is True.
    """
    def isLateralized(self):
        return True








"""
Conveniance class for un-lateralized blocks.
"""
class ProgInertBlock(ProgBlock):
    def isLateralized(self):
        return False






"""
A special case of ProgBlock with multiple paths.
"""
'''
class ProgMultiBlock(ProgBlock):
    """
    Returns a list of booleans (pass_list).
    Length of the pass_list is equal to the number of channels in the block.
    Channels that return True do not operate on their inputs, and simply pass them to the next block.
    """
    def getPassDescriptor(self):
        raise NotImplementedError

    def getNumChannels(self):
        raise NotImplementedError

'''




"""
Class that generates new ProgColumns using the method generateColumn.
The parentCols list will contain references to each parent column,
such that columns can access lateral outputs.
Additional information may be passed through the msg argument in
generateColumn and ProgNet.addColumn.
"""
class ProgColumnGenerator:
    def generateColumn(self, parentCols, msg = None):
        raise NotImplementedError






#============================<ProgColumn & ProgNet>============================#

"""
A column representing one sequential ANN with all of its lateral modules.
Outputs of the last forward run are stored for child column laterals.
Output of each layer is calculated as:
y = activation(block(x) + sum(laterals(x)))

colID -- A unique identifier for the column.
blockList -- A list of ProgBlocks that will be run sequentially.
parentCols -- A list of pointers to columns that will be laterally connectected.
              If the list is empty, the column is unlateralized.
"""
class ProgColumn(nn.Module):
    def __init__(self, colID, blockList, parentCols = []):
        super().__init__()
        self.colID = colID
        self.isFrozen = False
        self.parentCols = parentCols
        self.blocks = nn.ModuleList(blockList)
        self.numRows = len(blockList)
        self.lastOutputList = []

    def freeze(self, unfreeze = False):
        if not unfreeze:    # Freeze params.
            self.isFrozen = True
            for param in self.parameters():   param.requires_grad = False
        else:               # Unfreeze params.
            self.isFrozen = False
            for param in self.parameters():   param.requires_grad = True

    def getData(self):
        data = dict()
        data["colID"] = self.colID
        data["rows"] = self.numRows
        data["frozen"] = self.isFrozen
        #data["last_outputs"] = self.lastOutputList
        data["blocks"] = [block.getData() for block in self.blocks]
        data["parent_cols"] = [col.colID for col in self.parentCols]
        return data

    def forward(self, input):
        outputs = []
        x = input
        for r, block in enumerate(self.blocks):
            #if isinstance(block, ProgMultiBlock):
                #y = self.__forwardMulti(x, r, block)
            #else:
            y = self.__forwardSimple(x, r, block)
            outputs.append(y)
            x = y
        self.lastOutputList = outputs
        return outputs[-1]

    def __forwardSimple(self, x, row, block):
        currOutput = block.runBlock(x)
        if not block.isLateralized() or row == 0 or len(self.parentCols) < 1:
            y = block.runActivation(currOutput)
        elif isinstance(currOutput, list):
            for c, col in enumerate(self.parentCols):
                lats = block.runLateral(c, col.lastOutputList[row - 1])
                for i in range(len(currOutput)):
                    if currOutput[i] is not None and lats[i] is not None:
                        currOutput[i] += lats[i]
            y = block.runActivation(currOutput)
        else:
            for c, col in enumerate(self.parentCols):
                currOutput += block.runLateral(c, col.lastOutputList[row - 1])
            y = block.runActivation(currOutput)
        return y

    def __forwardMulti(self, x, row, block):
        if not isinstance(x, list):
            raise ValueError("[Doric]: Multiblock input must be a python list of inputs.")
        currOutput = block.runBlock(x)
        if not block.isLateralized() or row == 0 or len(self.parentCols) < 1:
            y = block.runActivation(currOutput)
        else:
            for c, col in enumerate(self.parentCols):
                lats = block.runLateral(c, col.lastOutputList[row - 1])
                for i, p in enumerate(block.getPassDescriptor()):
                    if not p:   currOutput[i] += lats[i]
            y = block.runActivation(currOutput)
        return y






"""
A progressive neural network as described in Progressive Neural Networks (Rusu et al.).
Columns can be added manually or with a ProgColumnGenerator.
https://arxiv.org/abs/1606.04671
"""
class ProgNet(nn.Module):
    def __init__(self, colGen = None):
        super().__init__()
        self.columns = nn.ModuleList()
        self.numRows = None
        self.numCols = 0
        self.colMap = dict()
        self.colGen = colGen

    def addColumn(self, col = None, msg = None):
        if not col:
            if self.colGen is None:
                raise ValueError("[Doric]: No column or generator supplied.")
            parents = [colRef for colRef in self.columns]
            col = self.colGen.generateColumn(parents, msg)
        self.columns.append(col)
        if col.colID in self.colMap:
            raise ValueError("[Doric]: Column ID must be unique.")
        self.colMap[col.colID] = self.numCols
        if self.numRows is None:
            self.numRows = col.numRows
        else:
            if self.numRows != col.numRows:
                raise ValueError("[Doric]: Each column must have equal number of rows.")
        self.numCols += 1
        return col.colID

    def freezeColumn(self, id):
        if id not in self.colMap:
            raise ValueError("[Doric]: No column with ID %s found." % str(id))
        col = self.columns[self.colMap[id]]
        col.freeze()

    def freezeAllColumns(self):
        for col in self.columns:
            col.freeze()

    def unfreezeColumn(self, id):
        if id not in self.colMap:
            raise ValueError("[Doric]: No column with ID %s found." % str(id))
        col = self.columns[self.colMap[id]]
        col.freeze(unfreeze = True)

    def unfreezeAllColumns(self):
        for col in self.columns:
            col.freeze(unfreeze = True)

    def isColumnFrozen(self, id):
        if id not in self.colMap:
            raise ValueError("[Doric]: No column with ID %s found." % str(id))
        col = self.columns[self.colMap[id]]
        return col.isFrozen

    def getColumn(self, id):
        if id not in self.colMap:
            raise ValueError("[Doric]: No column with ID %s found." % str(id))
        col = self.columns[self.colMap[id]]
        return col

    def forward(self, id, x):
        if self.numCols <= 0:
            raise ValueError("[Doric]: ProgNet cannot be run without at least one column.")
        if id not in self.colMap:
            raise ValueError("[Doric]: No column with ID %s found." % str(id))
        colToOutput = self.colMap[id]
        for i, col in enumerate(self.columns):
            y = col(x)
            if i == colToOutput:
                return y

    def getData(self):
        data = dict()
        data["cols"] = [c.getData() for c in self.columns]
        return data





#===============================================================================
