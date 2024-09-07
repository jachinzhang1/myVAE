import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(self, root, img_shape=(64, 64)) -> None:
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        self.filenames = sorted(os.listdir(root))
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path).convert('RGB')
        pipeline = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(self.img_shape),
            transforms.ToTensor()
        ])
        return pipeline(img)


def get_dataloader(root=None, batch_size=16, img_shape=(64, 64)):
    if root is None:
        raise FileExistsError
    dataset = CelebADataset(root=root, img_shape=img_shape)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    dataloader = get_dataloader(root='D:/Data/CelebA/Img/img_align_celeba_png')
    img = next(iter(dataloader))
    print(img.shape)
    N, C, H, W = img.shape
    assert N == 16
    img = torch.permute(img, (1, 0, 2, 3))
    img = torch.reshape(img, (C, 4, 4*H, W))
    img = torch.permute(img, (0, 2, 1, 3))
    img = torch.reshape(img, (C, 4*H, 4*W))
    img = transforms.ToPILImage()(img)
    img.save('./tmp.png')

