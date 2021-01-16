
# Imports.
import sys
import argparse

import torch

sys.path.append("../../../")

from Doric import ProgNet, ProgColumn, ProgColumnGenerator
from Doric import ProgConv2DBNBlock, ProgConv2DBlock


# Constants.
NAME_STR = "One-channel Conv Prognet Tester"
DESCRIP_STR = "A test script showing a one-channel convoloutional progressive neural network."


#---------------------------------[module code]---------------------------------


class ConvGenerator(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    def generateColumn(self, parentCols, msg = None):
        cols = []
        cols.append(ProgConv2DBNBlock(1, 10, 2, len(parentCols), layerArgs = {"padding": 1}))
        cols.append(ProgConv2DBNBlock(10, 20, 2, len(parentCols)))
        cols.append(ProgConv2DBlock(20, 20, 2, len(parentCols)))
        return ProgColumn(self.__genID(), cols, parentCols = parentCols)

    def __genID(self):
        id = self.ids
        self.ids += 1
        return id



def main(args):
    model = ProgNet(colGen=ConvGenerator())
    i1 = [[[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]]
    i2 = [[[1,1,1,1], [1,1,1,1], [0,0,0,0], [0,0,0,0]]]
    i3 = [[[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]]]
    z = torch.FloatTensor([i1, i2, i3])
    print()
    for ci in range(args.cols):
        print("Adding column %d." % ci)
        cid = model.addColumn()
        print("Data:")
        print(model.getData())
        print("Output from model([white, white_and_black, black]):")
        out = model(ci, z)
        for i in range(3):
            print("Input %d, channel 0:" % i)
            print(out[i][0])
            print("\n")
        print("\n\n")
        model.freezeAllColumns()







#--------------------------------[module setup]---------------------------------

'''
    Configures the given parser by adding arguments.
    Also defines the necessary inputs & default values of the module.
    If the script is being run directly, the function is used to config the main parser.
    If the module is being run as a sub-script, thee function can configure a subparser.
'''
def configCLIParser(parser):
    parser.add_argument("--cols", help="Columns to create and test.", type=int, default=4)
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = NAME_STR, description = DESCRIP_STR)   # Create module's cli parser.
    parser = configCLIParser(parser)
    args = parser.parse_args()
    main(args)

#===============================================================================
