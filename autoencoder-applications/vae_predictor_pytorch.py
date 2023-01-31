# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib
import gc

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

class Predictor(nn.Module):
    def __init__(self, latent_dim):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

def categorical_cross_entropy_loss_v2(y_pred, y_true):
    loss = F.binary_cross_entropy(y_pred, y_true, reduction='sum')
    return loss

def categorical_cross_entropy_loss(y_pred, y_true):
    loss = -torch.mean(torch.sum(y_true * torch.log(y_pred), dim=1))
    return loss

# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var, alpha, beta):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return alpha*BCE + beta*KLD, KLD


def train(train_loss_list,mean_latent_error,random_latent_loss,
        random_mean_1_latent_loss,train_recon_loss, train_kl_divergence_error,predictor_losses,
        epoch, alpha, beta, phi):
    vae.train()
    predictor.train()
    train_loss = 0
    predictor_loss = []
    latent_mse = nn.MSELoss()
    latent_errors = []
    recon_errors = []
    random_latent_error = []
    random_mean_1_latent_error = []
    kld_error = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        #data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae(data)
        y_pred = predictor(mu)
        y_pred = torch.argmax(y_pred, dim = 1).cpu().detach().numpy()
        y_pred = y_pred.astype(np.int64)
        y_pred = torch.Tensor(y_pred)
        print(y_pred.shape)
        print(y_pred)
        print(y_pred.dtype)
        print(label.dtype)
        print(label)
        #y_pred = y_pred.cpu().detach().numpy()
        #y_pred = torch.Tensor(y_pred)
        predictor_loss = categorical_cross_entropy_loss_v2(y_pred, label)
        _, encoded_mu_2, _ = vae(recon_batch)
        latent_error = latent_mse(mu, encoded_mu_2)
        latent_errors.append(phi*latent_error)
        recon_error = latent_mse(recon_batch, data.view(-1, 784))
        recon_errors.append(recon_error)
        loss, kld = loss_function(recon_batch, data, mu, log_var, alpha, beta)
        loss = loss + phi*predictor_loss
        ####
        random_sample = torch.randn(100, 2) #.cuda()
        random_decoded = vae.decoder(random_sample) #.cuda()
        random_encoded,_ = vae.encoder(random_decoded)
        random_latent_error.append(latent_mse(random_sample, random_encoded))

        random_sample_mean_1 = random_sample + 1
        random_decoded_mean_1 = vae.decoder(random_sample_mean_1) #.cuda()
        random_encoded_mean_1, _ = vae.encoder(random_decoded_mean_1)
        random_mean_1_latent_error.append(latent_mse(random_sample_mean_1, random_encoded_mean_1))
        ####
        loss.backward()
        train_loss += loss.item()
        kld_error += kld.item()
        predictor_loss += predictor_loss.item()
        optimizer.step()
        optimizer_predictor.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    mean_latent_error.append((sum(latent_errors)/len(latent_errors)).item())
    predictor_losses.append((sum(predictor_loss)/len(predictor_loss)).item())
    random_latent_loss.append((sum(random_latent_error)/len(random_latent_error)).item())
    random_mean_1_latent_loss.append((sum(random_mean_1_latent_error)/len(random_mean_1_latent_error)).item())
    train_loss_list.append(train_loss / len(train_loader.dataset))
    train_recon_loss.append((sum(recon_errors)/len(recon_errors)).item())
    train_kl_divergence_error.append(kld_error / len(train_loader.dataset))

def test(test_loss_list,test_mean_latent_error,test_recon_loss,test_kl_divergence_error,alpha, beta):
    vae.eval()
    latent_mse = nn.MSELoss()
    test_loss= 0
    latent_errors = []
    recon_errors = []
    kld_error = 0
    with torch.no_grad():
        for data, _ in test_loader:
            #data = data.cuda()
            recon, mu, log_var = vae(data)
            _, encoded_mu_2, _ = vae(recon)
            latent_error  = latent_mse(mu, encoded_mu_2)
            latent_errors.append(latent_error)
            recon_error = latent_mse(recon, data.view(-1,784))
            recon_errors.append(recon_error)
            # sum up batch loss
            t_error, kld = loss_function(recon, data, mu, log_var, alpha, beta)
            test_loss += t_error.item()
            kld_error += kld.item()

    test_loss /= len(test_loader.dataset)
    kld_error /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    test_mean_latent_error.append((sum(latent_errors)/len(latent_errors)).item())
    test_loss_list.append(test_loss)
    test_recon_loss.append((sum(recon_errors)/len(recon_errors)).item())
    test_kl_divergence_error.append(kld_error)
vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
predictor = Predictor(latent_dim = 2)

optimizer = optim.Adam(vae.parameters())
optimizer_predictor = optim.Adam(predictor.parameters())
alpha = [1]
beta = [1]
phi = [1,2,5,10,15,20]

for a in alpha:
    for b in beta:
        for p in phi:
            print('Experiment with alpha={}, beta={}, phi = {}'.format(a,b,p))
            vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
            predictor = Predictor(latent_dim = 2)

            optimizer = optim.Adam(vae.parameters())
            optimizer_predictor = optim.Adam(predictor.parameters())
            mean_latent_error = []
            test_mean_latent_error = []
            random_latent_loss = []
            random_mean_1_latent_loss = []
            test_loss_list = []
            train_loss_list = []
            train_recon_loss = []
            test_recon_loss = []
            train_kl_divergence_error = []
            test_kl_divergence_error = []
            predictor_losses = []
            for epoch in range(1, 100):
                train(train_loss_list,mean_latent_error,random_latent_loss,
                        random_mean_1_latent_loss,train_recon_loss,train_kl_divergence_error,
                         predictor_losses ,epoch,alpha=a,beta=b,phi=p)
                test(test_loss_list,test_mean_latent_error,test_recon_loss,
                        test_kl_divergence_error,alpha=a, beta=b)
                results = pd.DataFrame(
                                {
                                'mean_latent_error': mean_latent_error,
                                'test_mean_latent_error': test_mean_latent_error,
                                'random_latent_loss': random_latent_loss,
                                'random_mean_1_latent_loss': random_mean_1_latent_loss,
                                'train_loss': train_loss_list,
                                'train_predictor_loss': predictor_losses,
                                'test_loss': test_loss_list,
                                'train_reconstruction_mse_loss': train_recon_loss,
                                'test_reconstruction_mse_loss': test_recon_loss,
                                'train_kl_divergence_error': train_kl_divergence_error,
                                'test_kl_divergence_error': test_kl_divergence_error
                                })
                results_file_name = 'results/results_csv/results_alpha_{}_beta_{}_phi_{}.csv'.format(a,b,p)
                results.to_csv(results_file_name)
            path = './results/checkpoints/'
            checkpoint_name = 'model_alpha_{}_beta_{}_phi_{}.pth'.format(a,b,p)
            full_path = path + checkpoint_name
            torch.save(vae.state_dict(), full_path)
            del vae
