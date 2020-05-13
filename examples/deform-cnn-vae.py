import sys
import os
import torch
import torch.nn as nn
import itertools
import argparse

import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torch.utils import data

sys.path.append("../..")

from Doric import ProgNet, ProgColumn, ProgColumnGenerator
from Doric import ProgDenseBlock, ProgLambdaBlock, ProgInertBlock, ProgDeformConv2DBlock, ProgDeformConv2DBNBlock, ProgConvTranspose2DBNBlock

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", help="Specify whether the CPU should be used", type=bool, nargs='?', const=True, default=False)
parser.add_argument("--output", help="Specify where to log the output to", type=str, default="output")
parser.add_argument("--batch_size", help="Batch size", type=int, default=100)
parser.add_argument("--epochs", help="Epochs", type=int, default=50)
parser.add_argument("--lr", help="Epochs", type=float, default=0.0005)
args = parser.parse_args()

z_dim = 128

MODEL_SAVE_PATH = 'model'

writer = SummaryWriter()
device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')

# global variables to store the mean and variance
g_mu = None
g_var = None

mask_dataset = None
mask_loader = None
mask_loader_iter = None

"""
    Function called in train method to transform the input batch
"""
def transform_input(x, method):
    if method == 'denoise':
        return x + torch.randn(*x.shape) * 0.1
    elif method == 'colorize':
        img = x.mean(axis=1, keepdim=True)
        img = torch.cat((img, img, img), dim=1)
        return img
    elif method == 'inpaint':
        mask, _ = next(mask_loader_iter)
        return x * mask
    else:
        return x

def reparamaterize(x):
    mu, log_var = x

    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu

class ProgVariationalBlock(ProgInertBlock):
    def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module1 = nn.Linear(inSize, outSize)
        self.module2 = nn.Linear(inSize, outSize)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        mu = self.module1(x)
        var = self.module2(x)
        return (mu, var)

    def runLateral(self, i, x):
        raise NotImplementedError

    def runActivation(self, x):
        global g_mu
        global g_var
        mu, var = x
        g_mu = mu
        g_var = var
        return (mu, var)

    def getData(self):
        data = dict()
        data["type"] = "Variational"
        data["input_sizes"] = [self.inSize]
        data["output_sizes"] = [self.outSize, self.outSize]
        return data

    def getShape(self):
        return (self.inSize, self.outSize)

class VariationalAutoEncoderModelGenerator(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    def generateColumn(self, parentCols, msg = None):
        cols = []
        # Encode
        cols.append(ProgDeformConv2DBNBlock(3, 32, 3, 0, activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(ProgDeformConv2DBNBlock(32, 64, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(ProgDeformConv2DBNBlock(64, 128, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(ProgDeformConv2DBNBlock(128, 256, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(ProgDeformConv2DBNBlock(256, 512, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(ProgLambdaBlock(512, 512 * 4, lambda x: torch.flatten(x, start_dim=1)))

        cols.append(ProgVariationalBlock(512 * 4, z_dim, len(parentCols)))
        cols.append(ProgLambdaBlock(z_dim, z_dim, reparamaterize))

        # Decode
        cols.append(ProgDenseBlock(z_dim, 512 * 4, len(parentCols), activation=None))
        cols.append(ProgLambdaBlock(512 * 4, 512, lambda x: x.view(-1, 512, 2, 2)))
        cols.append(ProgConvTranspose2DBNBlock(512, 256, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(ProgConvTranspose2DBNBlock(256, 128, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(ProgConvTranspose2DBNBlock(128, 64, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(ProgConvTranspose2DBNBlock(64, 32, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(ProgConvTranspose2DBNBlock(32, 32, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(ProgDeformConv2DBlock(32, 3, 3, len(parentCols), activation=nn.Tanh(), layerArgs={'padding': 1}))
        
        return ProgColumn(self.__genID(), cols, parentCols = parentCols)

    def __genID(self):
        id = self.ids
        self.ids += 1
        return id

def train(model, batch_size, epochs, device, transform_method='none', skip_training=False, kl_weight=3e-5):
    global g_mu
    global g_var

    # create a new VAE column in our prognet
    col = model.addColumn()
    model = model.to(device)

    if skip_training:
        return col

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(64),
                                            transforms.ToTensor()])
    dataset = datasets.ImageFolder('../data/img_align_celeba', transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(epochs):
        for i, (x, _) in enumerate(data_loader):

            # Create a copy of the original (clean) batch
            with torch.no_grad():
                x_original = x.to(device)
            # Apply any transformation specified
            x = transform_input(x, transform_method)
            
            x = x.to(device)

            # Forward
            x_reconst = model(col, x)

            # MSE Loss
            reconst_loss = F.mse_loss(x_reconst, x_original)

            # KL Divergence
            kl_div = torch.mean(-0.5 * torch.sum(1 + g_var - g_mu ** 2 - g_var.exp(), dim = 1), dim = 0)

            loss = reconst_loss + kl_weight * kl_div

            # Backprop
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            
            if (i+1) % 10 == 0:
                print ("Col: {}, Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}, Loss: {:.4f}" 
                    .format(col, epoch+1, epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item(), loss.item()))

                # Tensorboard logging
                niter = epoch * len(data_loader) + i
                writer.add_scalar('Column {} Train/Reconst Loss'.format(col), reconst_loss.item(), niter)
                writer.add_scalar('Column {} Train/KL Div'.format(col), kl_div.item(), niter)
                writer.add_scalar('Column {} Train/Loss'.format(col), loss.item(), niter)
        
        with torch.no_grad():
            # Save the model output of the last batch as [original, augmented input, output]
            x_concat = torch.cat([x_original.cpu().data, x.cpu().data, x_reconst.cpu().data], dim=3)
            save_image(x_concat, os.path.join(args.output, 'col-{}-reconst-{}.png'.format(col, epoch+1)))
            writer.add_image("Column {}".format(col), torchvision.utils.make_grid(x_concat), epoch+1)

    torch.save(model.state_dict(), "{}-{}.pt".format(MODEL_SAVE_PATH, col))
    dataset = None
    data_loader = None
    return col

def forward(model, col, x):
    # TODO: Implement
    out = model(col, x)
    save_image(x, 'out.jpg')

if __name__ == "__main__":
    #model.load_state_dict(torch.load('model-3.pt'))
    model = ProgNet(colGen=VariationalAutoEncoderModelGenerator())

    # Col 1
    train(model, args.batch_size, args.epochs, device, transform_method='none')
    model.freezeAllColumns()
    # Col 2
    train(model, args.batch_size, args.epochs, device, transform_method='denoise')
    model.freezeAllColumns()
    # Col 3
    train(model, args.batch_size, args.epochs, device, transform_method='colorize')
    model.freezeAllColumns()
    mask_dataset = datasets.ImageFolder('../data/mask', transforms.ToTensor())
    mask_loader = torch.utils.data.DataLoader(mask_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    #Col 4
    mask_loader_iter = itertools.cycle(mask_loader)
    train(model, args.batch_size, args.epochs, device, transform_method='inpaint', kl_weight=3e-6)