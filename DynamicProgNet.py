

from .ProgNet import ProgNet



class DynamicProgNet(ProgNet):
    def __init__(self, colGen = None):
        super().__init__(colGen)
        self.latDict = dict()

    def addColumn(self, lateralCols = [], col = None, msg = None):
        latList = self.__buildLatList(lateralCols)
        if not col:
            assert self.colGen, "No column or generator supplied."
            col = self.colGen.generateColumn(latList, msg)
        if self.colShape is None:
            self.colShape = col.getShape()
        self.columns.append(col)
        assert not col.colID in self.colMap, "Column ID must be unique."
        self.colMap[col.colID] = self.numCols
        if self.numRows is None:
            self.numRows = col.numRows
        else:
            assert self.numRows == col.numRows, "Each column must have equal number of rows."
        self.latDict[col.colID] = latList
        self.numCols += 1
        return col.colID

    def forward(self, id, x):
        assert self.numCols > 0, "ProgNet cannot be run without at least one column."
        assert id in self.colMap, "Selected column has not been registered."
        colToOutput = self.colMap[id]
        lats = self.latDict[id]
        for i, col in enumerate(lats):
            y = col(x)
            if i == colToOutput:
                return y

    def __buildLatList(self, lateralCols, recurseRoot = True):
        latIDs = lateralCols
        latList = [self.getColumn(x) for x in latIDs]
        for latID in latIDs:
            middleIDs = self.latDict[latID]
            if middleIDs != []:
                lowerIDs, lowerLats = self.__buildLatList(middleIDs, recurseRoot = False)
                latList = latList + lowerLats
                latIDs = latIDs + lowerIDs
        if recurseRoot:
            return reversed(latIDs), reversed(latList)
        else:
            return latIDs, latList







#===============================================================================
