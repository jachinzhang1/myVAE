import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    VAE for 64x64 face generation. The hidden dimensions can be tuned.
    """
    def __init__(
            self,
            hiddens=[16, 32, 64, 128, 256],
            latent_dim=128,
    ) -> None:
        super().__init__()

        # encoder
        prev_channels = 3
        modules = []
        img_length = 64
        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, cur_channels, 3, 2, 1),
                    nn.BatchNorm2d(cur_channels),
                    nn.ReLU()
                )
            )
            prev_channels = cur_channels
            img_length //= 2
        self.encoder = nn.Sequential(*modules)
        self.mean_linear = nn.Linear(prev_channels * img_length * img_length, latent_dim)
        self.var_linear = nn.Linear(prev_channels * img_length * img_length, latent_dim)
        self.latent_dim = latent_dim

        # decoder
        modules = []
        self.decoder_projection = nn.Linear(latent_dim, prev_channels * img_length * img_length)
        self.decoder_input_chw = (prev_channels, img_length, img_length)
        for i in range(len(hiddens)-1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i],
                                       hiddens[i-1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hiddens[i-1]),
                    nn.ReLU()
                )
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0],
                                   hiddens[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hiddens[0]),
                nn.ReLU(),
                nn.Conv2d(hiddens[0], 3, 3, 1, 1),
                nn.ReLU(),
            )
        )
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, x):
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)
        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)
        eps = torch.rand_like(logvar)
        std = torch.exp(logvar / 2)
        z = mean + std * eps
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)

        return decoded, mean, logvar
    
    def sample(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        z = torch.randn(1, self.latent_dim).to(device)
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)
        return decoded


class VQVAE(nn.Module):
    """
    VQ-VAE for 64x64 face generation. The hidden dimensions can be tuned.
    """
    def __init__(
            self,
            hiddens=[16, 32, 64, 128, 256],
            latent_dim=128,
            embedding_dim=64,
            embedding_size=256,
    ) -> None:
        super().__init__()

        # encoder
        prev_channels = 3
        modules = []
        img_length = 64
        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, cur_channels, 3, 2, 1),
                    nn.BatchNorm2d(cur_channels),
                    nn.ReLU()
                )
            )
            prev_channels = cur_channels
            img_length //= 2
        self.encoder = nn.Sequential(*modules)
        self.mean_linear = nn.Linear(prev_channels * img_length * img_length, latent_dim)
        self.var_linear = nn.Linear(prev_channels * img_length * img_length, latent_dim)
        self.latent_dim = latent_dim

        # decoder
        modules = []
        prev_channels = hiddens[-1]
        for i in range(len(hiddens)-1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i],
                                       hiddens[i-1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hiddens[i-1]),
                    nn.ReLU()
                )
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0],
                                   hiddens[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hiddens[0]),
                nn.ReLU(),
                nn.Conv2d(hiddens[0], 3, 3, 1, 1),
                nn.ReLU(),
            )
        )
        self.decoder = nn.Sequential(*modules)

        # VQ-VAE
        # TODO: maybe there is some bug to be fixed
        self.embedding = nn.Embedding(embedding_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / embedding_size, 1.0 / embedding_size)
        self.embedding_size = embedding_size
        self.embedding_dim = embedding_dim
        self.embedding_projection = nn.Linear(latent_dim, embedding_dim)
        self.decoder_projection = nn.Linear(embedding_dim, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mean = self.mean_linear(x)
        logvar = self.var_linear(x)
        eps = torch.rand_like(logvar)
        std = torch.exp(logvar / 2)
        z = mean + std * eps
        z = self.embedding_projection(z)
        z = self.embedding(z)
        z = self.decoder_projection(z)
        z = torch.reshape(z, (-1, *self.decoder_input_chw))
        decoded = self.decoder(z)
        return decoded, mean, logvar

    def sample(self, batch_size, device):
        z = torch.randn(batch_size, self.latent_dim, device=device)
        z = self.embedding_projection(z)
        z = self.embedding(z)
        z = self.decoder_projection(z)
        z = torch.reshape(z, (-1, *self.decoder_input_chw))
        decoded = self.decoder(z)
        return decoded
    
class VQVAE2_0(VAE):
    # TODO: implement VQVAE2.0
    def __init__(self, 
                 latent_dim=128,
                 hiddens=[16, 32, 64, 128, 256],
                 embedding_size=512,
                 embedding_dim=64):
        super().__init__(hiddens, latent_dim)
        self.embedding = nn.Embedding(embedding_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / embedding_size, 1.0 / embedding_size)
        self.embedding_size = embedding_size
        self.embedding_dim = embedding_dim
        self.embedding_projection = nn.Linear(latent_dim, embedding_dim)
        self.decoder_projection = nn.Linear(embedding_dim, latent_dim)
        self.commitment_cost = 0.25

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mean = self.mean_linear(x)
        logvar = self.var_linear(x)
        eps = torch.rand_like(logvar)
        std = torch.exp(logvar / 2)
        z = mean + std * eps
        z = self.embedding_projection(z)
        z = self.embedding(z)
        z = self.decoder_projection(z)
        z = torch.reshape(z, (-1, *self.decoder_input_chw))
        decoded = self.decoder(z)
        return decoded, mean, logvar
    