
# Imports.
import sys
import argparse

import torch

sys.path.append("../../../")

from Doric import ProgNet, ProgColumn, ProgColumnGenerator
from Doric import ProgDenseBlock


# Constants.
NAME_STR = "One-channel FF Prognet Tester"
DESCRIP_STR = "A test script showing a simple one-channel feed-forward progressive neural network."


#---------------------------------[module code]---------------------------------


class FFGenerator(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    def generateColumn(self, parentCols, msg = None):
        cols = []
        cols.append(ProgDenseBlock(8, 20, len(parentCols)))
        cols.append(ProgDenseBlock(20, 50, len(parentCols)))
        cols.append(ProgDenseBlock(50, 10, len(parentCols)))
        cols.append(ProgDenseBlock(10, 2, len(parentCols)))
        return ProgColumn(self.__genID(), cols, parentCols = parentCols)

    def __genID(self):
        id = self.ids
        self.ids += 1
        return id



def main(args):
    model = ProgNet(colGen=FFGenerator())
    z = torch.FloatTensor([0,0,0,0,0,0,0,0])
    print()
    for ci in range(args.cols):
        print("Adding column %d." % ci)
        cid = model.addColumn()
        print("Data:")
        print(model.getData())
        print("Output from model(zeros):")
        print(model(ci, z))
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
