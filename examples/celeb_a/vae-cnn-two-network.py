import sys
import os
import torch
import torch.nn as nn

import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import save_image

sys.path.append("../../../")

from Doric import ProgNet, ProgColumn, ProgColumnGenerator
from Doric import ProgDenseBlock, ProgLambda, ProgInertBlock, ProgConv2DBlock, ProgConv2DBNBlock, ProgConvTranspose2DBNBlock

z_dim = 128
epochs = 1
batch_size = 144
learning_rate = 0.0005
KL_DIV_WEIGHT = 3e-5
samples_dir = 'output/vae-cnn-two-network'
num_samples = 144

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

g_mu = torch.cuda.FloatTensor().to(device)
g_var = torch.cuda.FloatTensor().to(device)


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

class EncoderModelGenerator(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    def generateColumn(self, parentCols, msg = None):

        # Encode
        b1 = ProgConv2DBNBlock(3, 32, 3, 0, activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1})
        b2 = ProgConv2DBNBlock(32, 64, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1})
        b3 = ProgConv2DBNBlock(64, 128, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1})
        b4 = ProgConv2DBNBlock(128, 256, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1})
        b5 = ProgConv2DBNBlock(256, 512, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1})
        b6 = ProgLambda(lambda x: torch.flatten(x, start_dim=1))

        b7 = ProgVariationalBlock(512 * 4, 128, len(parentCols))
        b8 = ProgLambda(reparamaterize)
        c = ProgColumn(self.__genID(), [b1, b2, b3, b4, b5, b6, b7, b8], parentCols = parentCols)
        return c

    def __genID(self):
        id = self.ids
        self.ids += 1
        return id

class DecoderModelGenerator(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    def generateColumn(self, parentCols, msg = None):
        b9 = ProgDenseBlock(128, 512 * 4, len(parentCols), activation=None)
        b10 = ProgLambda(lambda x: x.view(-1, 512, 2, 2))
        b11 = ProgConvTranspose2DBNBlock(512, 256, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1})
        b12 = ProgConvTranspose2DBNBlock(256, 128, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1})
        b13 = ProgConvTranspose2DBNBlock(128, 64, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1})
        b14 = ProgConvTranspose2DBNBlock(64, 32, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1})
        b15 = ProgConvTranspose2DBNBlock(32, 32, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1})
        b16 = ProgConv2DBlock(32, 3, 3, 0, activation=nn.Tanh(), layerArgs={'padding': 1})
        c = ProgColumn(self.__genID(), [b9, b10, b11, b12, b13, b14, b15, b16], parentCols = parentCols)
        return c

    def __genID(self):
        id = self.ids
        self.ids += 1
        return id

def train(encoder, decoder, epochs, data_loader, eoptimizer, doptimizer, col, device, add_noise=False):
    global g_mu
    global g_var
    for epoch in range(epochs):
        for i, (x, _) in enumerate(data_loader):
            # Forward pass

            if add_noise:
                x = x + torch.randn(*x.shape) * 0.7

            x = x.to(device)
            latent = encoder(col, x)
            x_reconst = decoder(col, latent)

            mu = g_mu
            log_var = g_var

            # KL Divergence
            #reconst_loss = F.mse_loss(x_reconst, x)
            t = x_reconst - x
            reconst_loss = 10.0 * t + torch.log(1. + torch.exp(- 2 * 10.0 * t)) - torch.log(torch.tensor(2.0))
            reconst_loss = (1. / 10.0) * reconst_loss.mean()

            kl_div = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            # Backprop and optimize
            #loss = reconst_loss + KL_DIV_WEIGHT * kl_div
            loss = reconst_loss + 1.0 * KL_DIV_WEIGHT * kl_div
            eoptimizer.zero_grad()
            doptimizer.zero_grad()
            loss.backward()
            eoptimizer.step()
            doptimizer.step()

            if (i+1) % 10 == 0:
                print ("Col: {}, Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}, Loss: {:.4f}"
                    .format(col, epoch+1, epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item(), loss.item()))

        with torch.no_grad():
            # Save the sampled images
            z = torch.randn(num_samples, z_dim)
            z = z.to(device)
            out = decoder(col, z)
            save_image(out, os.path.join(samples_dir, 'sampled/col-{}-sampled-{}.png'.format(col, epoch+1)))

            # Save the reconstructed images
            out = encoder(col, x)
            out = decoder(col, out)
            x_concat = torch.cat([x.cpu().data, x_reconst.cpu().data], dim=3)
            save_image(x_concat, os.path.join(samples_dir, 'reconst/col-{}-reconst-{}.png'.format(col, epoch+1)))

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(64),
                                            transforms.ToTensor()])

dataset = datasets.ImageFolder('data/img_align_celeba', transform)
train_loader1 = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

encoder = ProgNet(colGen=EncoderModelGenerator())
decoder = ProgNet(colGen=DecoderModelGenerator())
encoder = encoder.to(device)
decoder = decoder.to(device)

col1 = encoder.addColumn()
decoder.addColumn()

print(col1)

encoder = encoder.to(device)
decoder = decoder.to(device)

encoder_optimizer1 = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer1 = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

train(encoder, decoder, epochs, train_loader1, encoder_optimizer1, decoder_optimizer1, col1, device)

dataset2 = datasets.ImageFolder('data/img_align_celeba', transform)
train_loader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=True, drop_last=True)

encoder.freezeAllColumns()
decoder.freezeAllColumns()

col2 = encoder.addColumn()
decoder.addColumn()

print(col2)

encoder = encoder.to(device)
decoder = decoder.to(device)

encoder_optimizer2 = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer2 = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

train(encoder, decoder, epochs, train_loader2, encoder_optimizer2, decoder_optimizer2, col2, device, add_noise=True)
