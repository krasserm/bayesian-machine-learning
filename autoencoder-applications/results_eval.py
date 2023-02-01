import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import pandas as pd
import matplotlib
import gc
import sys
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import matplotlib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats  import multivariate_normal


gc.collect()
torch.cuda.empty_cache()
bs = 100
# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

def bce(recon_x, x):
    """
    used only for BCE
    """
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='mean')
    #KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE

def test(experiment_name):
    """
    prepare test data and calculate LRE of test instances as well as other metrics
    """
    vae.eval()
    i = 0
    latent_mse = nn.MSELoss()
    bce_loss =  nn.BCELoss()
    latent_errors = []
    recon_errors = []
    density_list = []
    labels = pd.DataFrame()
    encoded_data = pd.DataFrame()
    latent_reconstructed_data = pd.DataFrame()
    with torch.no_grad():
        for data, label in test_loader: ## change to test_loader
            #i = i + bs
            recon, mu, log_var = vae(data)
            _, encoded_mu_2, _ = vae(recon)
            latent_error  = latent_mse(mu, encoded_mu_2)
            latent_errors.append(latent_error)
            df_mu = pd.DataFrame(mu.cpu().numpy())
            df_mu_2 = pd.DataFrame(encoded_mu_2.cpu().numpy())
            encoded_data = pd.concat([encoded_data, df_mu])
            latent_reconstructed_data = pd.concat([latent_reconstructed_data, df_mu_2])
            df_label = pd.DataFrame(label.cpu().numpy())
            labels = pd.concat([labels, df_label])
            #if i >= 10000:
            #    break
    return encoded_data, latent_reconstructed_data, labels



def plot(encoded_data, label, filename):
    plt.scatter(encoded_data.iloc[:,0], encoded_data.iloc[:,1], c = label, label = 'Extrapolation')
    plt.legend(loc='upper left')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.xlim(-8,5)
    plt.ylim(-4.5,6)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
    model_path = (sys.argv)[1]
    print(model_path)
    experiment_name = model_path.rsplit('/',1)[1][6:-4]
    vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    encoded_data, latent_reconstructed_data, labels = test(experiment_name)
    filename = 'results/latent_space_vis/{}.png'.format(experiment_name)
    plot(encoded_data, labels, filename)
