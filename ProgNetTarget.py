
import copy
from .ProgNet import ProgNet


class ProgNetWithTarget(ProgNet):
    def __init__(self, colGen = None):
        super().__init__(colGen = colGen)
        self.targetColumns = []   # Should not be a ModuleList so params remain invisible to pytorch.

    def forwardTarget(self, id, x):
        assert self.numCols > 0, "ProgNet cannot be run without at least one column."
        assert id in self.colMap, "Selected column has not been registered."
        colToOutput = self.colMap[id]
        for i, col in enumerate(self.targetColumns):
            y = col(x)
            if i == colToOutput:
                return y

    def addColumn(self, col = None, msg = None):
        colID = super().addColumn(col = col, msg = msg)
        newTarget = copy.deepcopy(self.columns[self.colMap[colID]])
        newTarget.freeze()
        self.targetColumns.append(newTarget)
        self.synchTargets()
        return colID

    def synchTargets(self):
        for col, targ in zip(self.columns, self.targetColumns):
            targ.load_state_dict(col.state_dict())
            targ.freeze()


#===============================================================================
