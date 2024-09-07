import torch
from train import hyperparams
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader
import os
from time import time

from model import VAE
from dataset import get_dataloader


def reconstruct(device, dataloader, model: VAE, model_id, epochs_pretrained):
    print('Reconstructing...')
    if not os.path.exists('./reconstructed'):
        os.mkdir('./reconstructed')
    file_num = len(os.listdir('./reconstructed'))
    begin = time()
    model.eval()
    batch = next(iter(dataloader))
    x = batch[0:1, ...].to(device)
    output = model(x)[0]
    output = output[0].detach().cpu()
    input_img = batch[0].detach().cpu()
    combined = torch.cat((output, input_img), 1)
    img = ToPILImage()(combined)
    img.save('./reconstructed/%d-m%d-e%d.png' % (file_num + 1, model_id, epochs_pretrained))
    lasting = (time() - begin) * 1000
    print('Done in %.2f miliseconds.' % lasting)


def generate(device, model: VAE, model_id, epochs_pretrained):
    print('Generating...')
    if not os.path.exists('./generated'):
        os.mkdir('./generated')
    file_num = len(os.listdir('./generated'))
    begin = time()
    model.eval()
    output = model.sample(device)
    output = output[0].detach().cpu()
    img = ToPILImage()(output)
    img.save('./generated/%d-m%d-e%d.png' % (file_num + 1, model_id, epochs_pretrained))
    lasting = (time() - begin) * 1000
    print('Done in %.2f miliseconds.' % lasting)


if __name__ == '__main__':
    root='D:/Data/CelebA/Img/img_align_celeba_png'
    dataloader = get_dataloader(root, hyperparams['batch_size'])
    device = torch.device('cpu')
    
    model = VAE().to(device)
    model_id = 1
    epochs_trained = 15
    ckpt_path = './models/model-%d/epoch-%d.pth' % (model_id, epochs_trained)
    assert os.path.exists(ckpt_path), 'Model not found.'
    model.load_state_dict(torch.load(ckpt_path))
    reconstruct(device, dataloader, model, model_id, epochs_trained)
    generate(device, model, model_id, epochs_trained)