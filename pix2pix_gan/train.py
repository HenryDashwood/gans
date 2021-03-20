import os

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from typer import Typer

from model import Generator, Discriminator

app = Typer()


class FacadesDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w, d = img.shape
        w //= 2
        real_image = img[:, :w, :]
        input_image = img[:, w:, :]
        if self.transforms is not None:
            input_image = self.transforms(Image.fromarray(input_image))
            real_image = self.transforms(Image.fromarray(real_image))
        return input_image, real_image

    def __len__(self):
        return len(self.imgs)


def train(
    generator,
    discriminator,
    num_epochs: int,
    batch_size: int,
    lr: int,
    K: int,
    image_width: int,
    image_height: int,
    device: torch.device,
):
    tb_writer = SummaryWriter()

    discriminator_optimiser = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
    )
    generator_optimiser = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=(0.5, 0.9)
    )

    if os.path.exists("checkpoints/pix2pix-optim-d.pytorch"):
        discriminator_optimiser.load_state_dict(
            torch.load("checkpoints/pix2pix-optim-d.pytorch")
        )
    if os.path.exists("checkpoints/pix2pix-optim-g.pytorch"):
        generator_optimiser.load_state_dict(
            torch.load("checkpoints/pix2pix-optim-g.pytorch")
        )

    Lambda = 100
    bce_loss = nn.BCELoss()
    mae_loss = nn.L1Loss()

    real_labels = torch.ones(batch_size, 1, 30, 30).to(device)
    fake_labels = torch.zeros(batch_size, 1, 30, 30).to(device)

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((268, 268)),
            torchvision.transforms.RandomCrop((image_width, image_height)),
            torchvision.transforms.Lambda(lambda img: (np.array(img) / 127.5) - 1),
            torchvision.transforms.ToTensor(),
        ]
    )

    train_set = FacadesDataset(root="data/facades/train", transforms=transforms)
    val_set = FacadesDataset(root="data/facades/val", transforms=transforms)
    test_set = FacadesDataset(root="data/facades/test", transforms=transforms)

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=0)

    fixed_inputs, _ = next(iter(train_loader))
    fixed_inputs = fixed_inputs.to(device).float()

    num_steps = len(train_set) // batch_size
    for epoch in range(num_epochs):
        d_loss, g_loss = 0, 0
        for i, (src_images, tgt_images) in enumerate(train_loader):
            if i == num_steps:
                break
            for k in range(K):
                src_images = src_images.to(device).float()
                tgt_images = tgt_images.to(device).float()
                fake_images = generator(src_images)
                discriminator_optimiser.zero_grad()
                real_outputs = discriminator(torch.cat((src_images, tgt_images), 1))
                fake_outputs = discriminator(torch.cat((src_images, fake_images), 1))
                d_x = bce_loss(real_outputs, real_labels)
                d_g_z = bce_loss(fake_outputs, fake_labels)
                loss = d_x + d_g_z
                loss.backward()
                discriminator_optimiser.step()
                d_loss += loss
            d_loss /= 2
            src_images = src_images.to(device).float()
            tgt_images = tgt_images.to(device).float()
            generator.zero_grad()
            fake_images = generator(src_images)
            outputs = discriminator(torch.cat((src_images, fake_images), 1))
            loss_a = bce_loss(outputs, real_labels)
            loss_b = mae_loss(fake_images, tgt_images)
            loss = loss_a + (Lambda * loss_b)
            loss.backward()
            generator_optimiser.step()
            g_loss += loss
        if epoch % 10 == 0:
            print(f"E:{epoch} G Loss:{g_loss / num_steps} D Loss:{d_loss / num_steps}")
            torch.save(
                discriminator_optimiser.state_dict(),
                "checkpoints/pix2pix-optim-d.pytorch",
            )
            torch.save(
                generator_optimiser.state_dict(), "checkpoints/pix2pix-optim-g.pytorch"
            )
            torch.save(
                discriminator.state_dict(), "checkpoints/pix2pix-network-d.pytorch"
            )
            torch.save(generator.state_dict(), "checkpoints/pix2pix-network-g.pytorch")
            generated = generator(fixed_inputs).detach().cpu()
            grid = torchvision.utils.make_grid(
                generated, nrow=5, padding=10, pad_value=1, normalize=True
            )
            tb_writer.add_image("Pix2Pix/Output", grid, epoch)


@app.command()
def main(
    num_epochs: int = 11,
    batch_size: int = 64,
    K: int = 2,
    lr: float = 2e-4,
    image_width: int = 256,
    image_height: int = 256,
    device: str = "cuda",
):
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    if os.path.exists("checkpoints/pix2pix-network-d.pytorch"):
        discriminator.load_state_dict(
            torch.load("checkpoints/pix2pix-network-d.pytorch")
        )
    if os.path.exists("checkpoints/pix2pix-network-g.pytorch"):
        generator.load_state_dict(torch.load("checkpoints/pix2pix-network-g.pytorch"))

    train(
        generator,
        discriminator,
        num_epochs,
        batch_size,
        lr,
        K,
        image_width,
        image_height,
        device,
    )


if __name__ == "__main__":
    app()