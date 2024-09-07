TODO: accomplish readme

# Variational Autoencoder Demo

## Instruction

This is a demo of Variational Autoencoder (VAE) using PyTorch. The demo is based on the [VAE tutorial](https://zhouyifan.net/2022/12/19/20221016-VAE/).

## Project Contents

```
./
├── requirements.txt
├── model.py
├── README.md
├── train.py
└── test.py
```

- `model.py`: Define the VAE model.
- `train.py`: Train and save the VAE model. Models are saved at `./models/model-x/epoch-x.pth`.
- `test.py`: Reconstruct and generate samples using the pretrained VAE model.

## Requirements

```
Pillow==10.4.0
torch==2.3.1+cu121
torchvision==0.19.0
torchvision==0.18.1+cu121
tqdm==4.65.0
```

## Model Architecture

The VAE model consists of an encoder and a decoder. The encoder is a convolutional neural network (CNN) that maps the input image to a latent space. The decoder is also a CNN that maps the latent space back to the input image. The VAE model also includes a mean and a variance linear layer to model the distribution of the latent space.

```
VAE(
  (encoder): Sequential(
    (0): Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (1): Sequential(
      (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (2): Sequential(
      (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (3): Sequential(
      (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (4): Sequential(
      (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (mean_linear): Linear(in_features=1024, out_features=128, bias=True)
  (var_linear): Linear(in_features=1024, out_features=128, bias=True)
  (decoder_projection): Linear(in_features=128, out_features=1024, bias=True)
  (decoder): Sequential(
    (0): Sequential(
      (0): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))        
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (1): Sequential(
      (0): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (2): Sequential(
      (0): ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (3): Sequential(
      (0): ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (4): Sequential(
      (0): ConvTranspose2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): ReLU()
    )
  )
)
```

## Usage

### Download dataset

Download the dataset from [here](https://pan.baidu.com/s/1eSNpdRG#list/path=%2F)(password: rp0s) and extract it to your data directory.

### Install requirements

Install the required packages by running:

```bash
pip install -r requirements.txt
```

### Train and Test the VAE

Run `train.py` to train the VAE and run `test.py` to reconstruct and generate samples:

## Results

The test results of the VAE are saved at `./reconstructed` and `./generated`.

This is a reconstruction from a sample in the dataset:

![reconstructed](./assets/reconstructed.png)

This is a generation from Gaussian noise figure by using the pretrained VAE model:

![generated](./assets/generated.png)


