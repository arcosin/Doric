import sys
import os
import torch
import torch.nn as nn

import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image

sys.path.append("../")

from Doric import ProgNet, ProgColumn, ProgColumnGenerator
from Doric import ProgDenseBlock, ProgLambdaBlock, ProgInertBlock

image_size = 784
h_dim = 400
z_dim = 20
epochs = 15
batch_size = 128
learning_rate = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

g_mu = torch.cuda.FloatTensor().to(device)
g_var = torch.cuda.FloatTensor().to(device)


def reparamaterize(x):
    mu, log_var = x
    std = torch.exp(log_var/2)
    eps = torch.randn_like(std)
    return mu + eps * std

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

class AutoEncoderModelGenerator(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    def generateColumn(self, parentCols, msg = None):
        b1 = ProgDenseBlock(784, 100, 0)
        b2 = ProgDenseBlock(100, 30, len(parentCols))
        b3 = ProgVariationalBlock(30, 15, len(parentCols))

        b4 = ProgLambdaBlock(15, 15, reparamaterize)

        b5 = ProgDenseBlock(15, 30, len(parentCols))
        b6 = ProgDenseBlock(30, 100, len(parentCols))
        b7 = ProgDenseBlock(100, 784, len(parentCols), activation=nn.Sigmoid())
        c = ProgColumn(self.__genID(), [b1, b2, b3, b4, b5, b6, b7], parentCols = parentCols)
        return c
    def __genID(self):
        id = self.ids
        self.ids += 1
        return id

def train(model, epochs, data_loader, optimizer, col, device):
    global g_mu
    global g_var
    for epoch in range(epochs):
        for i, (x, _) in enumerate(data_loader):
            # Forward pass
            x = x.to(device).view(-1, image_size)
            x_reconst = model(col, x)

            mu = g_mu
            log_var = g_var
            # Compute reconstruction loss and kl divergence
            # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Backprop and optimize
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print ("Col: {}, Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                    .format(col, epoch+1, epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))
        
        with torch.no_grad():
            ## Save the sampled images
            #z = torch.randn(batch_size, z_dim).to(device)
            #out = model.decode(z).view(-1, 1, 28, 28)
            #save_image(out, os.path.join('samples', 'sampled-{}.png'.format(epoch+1)))
        
            # Save the reconstructed images
            out = model(col, x)
            x_concat = out.view(-1, 1, 28, 28)
            save_image(x_concat, os.path.join('samples', 'col-{}-reconst-{}.png'.format(col, epoch+1)))

mnist_dataset = torchvision.datasets.MNIST(root='./data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# MNIST Data loader
mnist_data_loader = torch.utils.data.DataLoader(dataset=mnist_dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

# EMNIST-letters need to be flipped and rotated
emnist_transforms = transforms.Compose([
    lambda img: img.rotate(-90, expand = 1),
    lambda img: transforms.functional.hflip(img),
    transforms.ToTensor()
])

emnist_dataset = torchvision.datasets.EMNIST(root='./data',
                                     train=True,
                                     split='letters',
                                     transform=emnist_transforms,
                                     download=True)

# EMNIST Data loader
emnist_data_loader = torch.utils.data.DataLoader(dataset=emnist_dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

model = ProgNet(colGen=AutoEncoderModelGenerator())
model = model.to(device)

mnist_col = model.addColumn()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(model, epochs, mnist_data_loader, optimizer, mnist_col, device)

model.freezeAllColumns()

emnist_col = model.addColumn()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(model, epochs, emnist_data_loader, optimizer, emnist_col, device)