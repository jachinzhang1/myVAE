import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from torch.utils.tensorboard import SummaryWriter

from model import VAE, VQVAE, VQVAE2_0
from dataset import get_dataloader


hyperparams = {
    'epochs': 20,
    'batch_size': 512,
    'kl_weight': 0.00025,
    'lr': 1e-3,
    'warm_start': False,
    'pretrained_model_id': 1,
    'pretrained_epochs': 13,
}
writer = SummaryWriter()

def loss_fn(y, y_hat, mean, logvar):
    # print(y_hat.size(), y.size())
    assert y_hat.size() == y.size()
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
    loss = recons_loss + hyperparams['kl_weight'] * kl_loss
    return loss


def train(device, dataloader: DataLoader, model, pre_epoch_num: int):
    print('Training...')
    if pre_epoch_num >= hyperparams['epochs']:
        print('Pretrained model has been trained for %d epochs, which is more than the required epochs %d. Training will be skipped.' % (pre_epoch_num, hyperparams['epochs']))
        return
    
    optimizer = optim.Adam(model.parameters(), hyperparams['lr'])
    dataset_len = len(dataloader.dataset)
    # TODO: lr decay
    
    if not os.path.exists('./models'):
        os.makedirs('./models', exist_ok=True)
    model_num = len(os.listdir('./models')) + int(not hyperparams['warm_start'])

    # train loop
    for epoch in range(hyperparams['epochs']):
        # print('Epoch %d/%d' % (epoch + 1, hyperparams['epochs']))
        processBar = tqdm(dataloader, unit='batch')
        model.train(True)
        loss_sum = 0
        processBar.set_description('Epoch %d/%d' % (pre_epoch_num + epoch + 1, hyperparams['epochs']))
        for idx, x in enumerate(processBar):
            x = x.to(device)
            y_hat, mean, logvar = model(x)
            loss = loss_fn(x, y_hat, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            writer.add_scalar('Training Loss', loss.item(), (pre_epoch_num + epoch) * len(processBar) + idx)

        loss_avg = loss_sum / dataset_len
        writer.add_scalar('Epoch Loss', loss_avg, pre_epoch_num + epoch + 1)
        # print(f'Epoch {epoch + 1} Loss: {loss_avg:.4f}')
        processBar.set_description('Epoch %d/%d Loss: %.4f' % (pre_epoch_num + epoch + 1, hyperparams['epochs'], loss_avg))
        processBar.close()
        
        # save model
        if not os.path.exists('./models/model-%d' % model_num):
            os.mkdir('./models/model-%d' % model_num)
        torch.save(model.state_dict(), './models/model-%d/epoch-%d.pth' % (model_num, pre_epoch_num + epoch + 1))
        print('Model saved.')

        if pre_epoch_num + epoch + 1 == hyperparams['epochs']:
            print('Training finished.')
            break


def main():
    root='D:/Data/CelebA/Img/img_align_celeba_png'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = get_dataloader(root, hyperparams['batch_size'])
    model = VAE().to(device)
    # model = VQVAE2_0().to(device)
    
    pre_epoch_num = 0
    if hyperparams['warm_start']:
        ckpt_path = './models/model-%d/epoch-%d.pth' % (hyperparams['pretrained_model_id'], hyperparams['pretrained_epochs'])
        model.load_state_dict(torch.load(ckpt_path))
        pre_epoch_num = int(ckpt_path.split('/')[-1].split('-')[-1].split('.')[0])
    
    batch = next(iter(dataloader))
    dummy_input = batch[0:1, ...].to(device)
    with SummaryWriter(comment='VAE') as w:
        w.add_graph(model, (dummy_input,))
    train(device, dataloader, model, pre_epoch_num)


if __name__ == '__main__':
    main()