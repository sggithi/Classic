import torch
import torch.nn as nn
import torchvision
import numpy as np
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

class Encoder(nn.Module):
    def __init__(self,in_channels=1, hidden_dims = [16,32,64], latent_dim=2,
              model_name='VAE'):
        """
        Initialize the encoder model.

        Args:
            in_channels : number of channels of input image
            hidden_dims : list of hidden layer dimensions
            latent_dim : dimension of latent vector
            model_name : type of model (beta-VAE or AE)
        """
        super(Encoder, self).__init__()
        
        self.hidden_dims = hidden_dims
        self.model_name = model_name
        self.model = None
        self.fc_mean = None
        self.fc_logvar = None
        
        layers = []
        layers.append(nn.Conv2d(in_channels, hidden_dims[0], kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(negative_slope=0.01))
        
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Conv2d(hidden_dims[i - 1], hidden_dims[i], kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(negative_slope=0.01))

        self.model = nn.Sequential(*layers)

        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim, bias=True)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim, bias=True)    
        

    def reparametrize(self, mu, logvar, eps):
        """
        Returns the reparametrized latent vector.

        Args:
            mu : latent mean (type : torch.Tensor, size : (batch_size, latent_dim))
            logvar : latent log variance (type : torch.Tensor, size : (batch_size, latent_dim))
            eps : random noise for encoder (type : torch.Tensor, size : (batch_size, latent_dim))
        Returns:
            rp : reparametrized latent vector (type : torch.Tensor, size : (batch_size, latent_dim))
        """
        rp = None

        var = torch.exp(logvar)
        sigma = torch.exp(logvar * 0.5)
        rp = mu + sigma * eps

        return rp
    
    def sample_noise(self, logvar):
        return torch.randn_like(logvar)
    
    def forward(self, x, eps=None):
        """
        Forward pass of the encoder.

        Args:
            x : the input to the encoder (image) (type : torch.Tensor, size : (batch_size, 1, 16, 16))
            eps : random noise for encoder (type : torch.Tensor, size : (batch_size, latent_dim))
        Returns:
            For VAE, return mu, logvar, rp
                mu : latent mean (type : torch.Tensor, size : (batch_size, latent_dim))
                logvar : latent log variance (type : torch.Tensor, size : (batch_size, latent_dim))
                rp : reparametrized latent vector (type : torch.Tensor, size : (batch_size, latent_dim))
            For AE, return out
                out : latent vector (type : torch.Tensor, size : (batch_size, latent_dim))
        """
        rp = None
        out = None
        
        # input -> Conv Layer 통과 256 -> -> hiddens[-1] 사이즈
        x = self.model(x)
        # 한 줄로 펴기
        x = x.view(x.size(0), -1)
        
        if 'VAE' in self.model_name:
            mu = self.fc_mean(x)
            logvar = self.fc_logvar(x)
            if eps is None:
              eps = self.sample_noise(logvar)
            rp = self.reparametrize(mu, logvar, eps)
            
            return mu, logvar, rp
        else:
            out = self.fc_mean(x)
            return out


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims = [16,32,64],expand_dim=4):
        """
        Initialize the decoder model.

        Args:
            latent_dim : dimension of latent vector (type : int)
            hidden_dims : list of hidden layer dimensions (type : list, default : None)
                - reverse order of encoder hidden layer dimensions
            expand_dim : size of the first hidden layer input of self.decoder (type : int)
                - the first hidden layer input of self.decoder is (B, self.hidden_dims[-1], self.expand_dim, self.expand_dim)
        """
        super(Decoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.expand_dim = expand_dim
        self.input_layer = None
        self.decoder = None
        self.last_layer = None
        
        self.input_layer = nn.Linear(latent_dim, self.hidden_dims[-1] * (self.expand_dim ** 2))
        
        decoder_layers = []
        
        for i in range(len(hidden_dims) - 2, -1, -1):
            decoder_layers.append(nn.ConvTranspose2d(hidden_dims[i + 1], hidden_dims[i], kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.LeakyReLU(negative_slope=0.01))
         
        self.decoder = nn.Sequential(*decoder_layers)
        
        last_layers = []
        last_layers.append(nn.ConvTranspose2d(hidden_dims[0], 1, kernel_size=3, stride=1, padding=1))
        last_layers.append(nn.Sigmoid())
        
        self.last_layer = nn.Sequential(*last_layers)
        

    def forward(self, x):
        """
        Foward pass of the decoder.

        Args:
            x : the input to the decoder (latent vector) (type : torch.Tensor, size : (batch_size, latent_dim))
        Returns:
            out : the output of the decoder (type : torch.Tensor, size : (batch_size, 1, 16, 16))
        """
        out = None
 
        x = self.input_layer(x)  # self.hidden_dims[-1] * (self.expand_dim ** 2)
        
        # (batch_size, self.hidden_dims[-1], self.expand_dim, self.expand_dim)
        x = x.view(x.size(0), self.hidden_dims[-1], self.expand_dim, self.expand_dim)
        x = self.decoder(x)
        x = self.last_layer(x)
        out = x

        return out

def reconstruction_loss(recon_x, x):
    """
    Returns the reconstruction loss of VAE.
    
    Args:
        recon_x : reconstructed x (type : torch.Tensor, size : (batch_size, 1, 16, 16)
        x : original x (type : torch.Tensor, size : (batch_size, 1, 16, 16)
    Returns:
        recon_loss : reconstruction loss (type : torch.Tensor, size : (1,))
    """
    loss = 0.0
    eps = 1e-18
    batch_size = x.size(0)

    recon_x_flat = recon_x.view(-1)
    x_flat = x.view(-1)
    loss = -torch.sum(x_flat * torch.log(recon_x_flat + eps) + (1 - x_flat) * torch.log(1 - recon_x_flat + eps))

    loss = loss / batch_size
    return loss

def KLD_loss(mu, logvar):
    """
    Returns the regularization loss of VAE.
    
    Args:
        mu : latent mean (type : torch.Tensor, size : (batch_size, latent_dim))
        logvar : latent log variance (type : torch.Tensor, size : (batch_size, latent_dim))
    Returns:
        kld_loss : regularization loss (type : torch.Tensor, size : (1,))
    """
    batch_size = mu.size(0)
    kld_loss = None

    kld_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) -logvar.exp())

    kld_loss = kld_loss / batch_size
    return kld_loss

def loss_function(recon_x, x, mu, logvar,beta=1,return_info=False):
    """
    Returns the loss of beta-VAE.

    Args:
        recon_x : reconstructed x (type : torch.Tensor, size : (batch_size, 1, 16, 16)
        x : original x (type : torch.Tensor, size : (batch_size, 1, 16, 16)
        mu : latent mean (type : torch.Tensor, size : (batch_size, latent_dim))
        logvar : latent log variance (type : torch.Tensor, size : (batch_size, latent_dim))
        beta : beta value for beta-VAE (type : float)
    Returns:
        loss : loss of beta-VAE (type : torch.Tensor, size : (1,))
            - Reconstruction loss + beta * Regularization loss
            Recon_loss : reconstruction loss
               kld_loss : KL divergence loss
    """    
    Recon_loss = reconstruction_loss(recon_x, x)
    kld_loss = KLD_loss(mu, logvar)

    loss = Recon_loss + beta * kld_loss

    if return_info:
        return {"loss" : loss,
                "recon_loss" : Recon_loss,
                "kld_loss" : kld_loss}
    else :
        return loss

class dataloader(torch.utils.data.Dataset):
    def __init__(self,train=True, batch_size = 64):
        """"
        Initialize the dataloader class.

        Args:
            train : whether to use training dataset or test dataset (type : bool)
            batch_size : how many samples per batch to load (type : int)
        """
        super(dataloader, self).__init__()
        self.batch_size = batch_size
        self.transform = None
  

        self.dataset = None
   
        self.dataset = torchvision.datasets.MNIST (root='./data', train=train, transform=self.transform, download=True)

        self.dataloader = None

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=train, drop_last=True)

    def __len__(self):
        return len(self.dataloader)
    def __iter__(self):
        return iter(self.dataloader)
    def __getitem__(self, idx):
        return self.dataloader[idx]

class training_VAE:
    def __init__(self, train_loader, test_loader, encoder, decoder, device,
                 config, save_img = False, model_name='VAE',beta=1, img_show=False):
        """"
        Initialize the training_VAE class.

        Args:
            train_loader : the dataloader for training dataset (type : torch.utils.data.DataLoader)
            test_loader : the dataloader for test dataset (type : torch.utils.data.DataLoader)
            encoder : the encoder model (type : Encoder)
            decoder : the decoder model (type : Decoder)
            device : the device where the model will be trained (type : torch.device)
            config : the configuration for training (type : SimpleNamespace)
            save_img : whether to save the generated images during training
            model_name : type of model - VAE or AE
                - VAE includes VAE and beta-VAE
            beta : beta value for beta-VAE (type : float)
            img_show : whether to show the generated images during training
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = config.epoch
        self.lr = config.lr
        self.latent_dim = config.latent_dim
        self.batch_size = config.batch_size
        self.device = device

        self.generated_img = []
        self.Recon_loss_history = []
        self.KLD_loss_history = []
        self.system_info = getSystemInfo()
        self.save_img = save_img
        self.model_name = model_name
        if self.model_name == 'beta_VAE':
            self.model_name = f"{self.model_name}_{beta}"
        self.beta = beta
        self.img_show = img_show

        self.encoder = encoder
        self.decoder = decoder

        self.optimizer = None
 
        self.optimizer =  torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=self.lr)


    def make_gif(self):
        """
        Save the generated images as a gif file.
        """
        if len(self.generated_img) <= 1:
            print("No frame to save")
            return
        else :
            print("Saving gif file...")
            for i in range(len(self.generated_img)):
                self.generated_img[i] = Image.fromarray(self.generated_img[i])
            self.generated_img[0].save(f"./{self.model_name}_generated_img.gif",
                                save_all=True, append_images=self.generated_img[1:], 
                                optimize=False, duration=700, loop=1) 
    def one_iter_train(self, images,label, eps):
        """
        Train the model for one iteration.

        Args:
            
            images : the input images (type : torch.Tensor, size : (batch_size, 1, 16, 16))
            label : the input labels (type : torch.Tensor, size : (batch_size))
            eps : random noise for encoder (type : torch.Tensor, size : (batch_size, latent_dim))
                - it is used in reparametrization trick
        Returns:
            dict : dictionary of losses
                - VAE :
                recon_loss : the reconstruction loss of the model (type : torch.Tensor, size : (1,))
                kld_loss : the regularization (KL divergence) loss of the model (type : torch.Tensor, size : (1,))
                - AE :
                recon_loss : the reconstruction loss of the model (type : torch.Tensor, size : (1,))
        """
        recon_loss = None
        kld_loss = None
        loss = 0.0

        total_loss = None
        self.optimizer.zero_grad()        
        if self.model_name == 'AE':
            out = self.encoder.forward(images)
            x_from_z =self.decoder.forward(out)
            recon_loss = reconstruction_loss(x_from_z, images)
            kld_loss = torch.empty((1,))
            total_loss = recon_loss
        elif self.model_name == 'VAE':
            mu, logvar, rp = self.encoder.forward(images, eps)
            x_from_z = self.decoder.forward(rp)
            results = loss_function(x_from_z, images, mu, logvar, self.beta, return_info=True)
            recon_loss, kld_loss = results['recon_loss'], results['kld_loss']
            total_loss = results['loss']
        else:
            mu, logvar, rp = self.encoder.forward(images, eps)   
            x_from_z = self.decoder.forward(rp) 
            results = loss_function(x_from_z, images, mu, logvar, self.beta, return_info=True)
            recon_loss, kld_loss = results['recon_loss'], results['kld_loss']
            total_loss = results['loss']

        total_loss.backward()
        self.optimizer.step()


        return {
                "recon_loss" : recon_loss.item(),
                "kld_loss" : kld_loss.item()
                }
    def get_fake_images(self, image,labels,eps):
        self.encoder.eval()
        self.decoder.eval()
        if self.model_name == 'AE':
            with torch.no_grad():
                rp = self.encoder(image)
                fake_images = self.decoder(rp)
        elif 'VAE' in self.model_name:
            with torch.no_grad():
                mean, logvar, rp = self.encoder(image,eps)
                fake_images = self.decoder(rp)
        else:
            raise NotImplementedError(f"Please choose the model type in ['VAE', 'AE'], not {self.model_name}")
        return fake_images
    def train(self):
        """
        Train the VAE model.
        """
        try : 
            for epoch in range(1,self.num_epochs+1):
                pbar = tqdm(enumerate(self.train_loader,start=1), total=len(self.train_loader))
                for i, (images, labels) in pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    eps = torch.randn(self.batch_size, self.latent_dim).to(self.device)
                    if epoch == 1 and i == 1:
                        fake_images = self.get_fake_images(images,labels,eps)
                        grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True).detach().cpu().permute(1,2,0).numpy()
                        self.generated_img.append((grid_img* 255).astype('uint8'))
                    self.encoder.train()
                    self.decoder.train()
                    
                    results = self.one_iter_train(images,labels,eps)
                    self.encoder.eval()
                    self.decoder.eval()
                    recon_loss, kld_loss = results['recon_loss'], results['kld_loss']
                    self.Recon_loss_history.append(recon_loss)
                    self.KLD_loss_history.append(kld_loss)
                    
                    fake_images = self.get_fake_images(images, labels,eps)

                    pbar.set_description(
                        f"Epoch [{epoch}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], Total loss : {recon_loss + kld_loss:.6f} Recon Loss: {recon_loss:.6f}, KLD Loss: {kld_loss:.6f}")
                
                grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True).detach().cpu().permute(1,2,0).numpy()
                self.generated_img.append((grid_img* 255).astype('uint8'))
                if self.img_show:
                    plt.imshow(grid_img)
                    plt.pause(0.01)
  
        except KeyboardInterrupt:
            print('Keyboard Interrupted, finishing training...')
        if self.save_img:
            self.make_gif() 
        
        return {'encoder' : self.encoder,
                'encoder_state_dict' : self.encoder.state_dict(),
                'decoder' : self.decoder,
                'decoder_state_dict' : self.decoder.state_dict(),
                'Recon_loss_history' : self.Recon_loss_history,
                'generated_img' : self.generated_img[-1],
                'KLD_loss_history' : self.KLD_loss_history}