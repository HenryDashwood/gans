import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from typer import Typer

from model import Generator, Discriminator

app = Typer()


class FacadesDataset(object):
    def __init__(self, root, transforms, category):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(root)))
        self.category = category

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w, d = img.shape
        w //= 2
        if self.category == 0:
            input_image = img[:, w:, :]
            if self.transforms is not None:
                input_image = self.transforms(Image.fromarray(input_image))
            return input_image
        else:
            real_image = img[:, :w, :]
            if self.transforms is not None:
                real_image = self.transforms(Image.fromarray(real_image))
            return real_image

    def __len__(self):
        return len(self.imgs)


def train(
    netDX,
    netDY,
    netG,
    netF,
    num_epochs: int,
    batch_size: int,
    lr: float,
    K: int,
    image_width: int,
    image_height: int,
    device,
):
    tb_writer = SummaryWriter()

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((268, 268)),
            torchvision.transforms.RandomCrop((image_width, image_height)),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.Lambda(lambda img: (np.array(img) / 127.5) - 1),
            torchvision.transforms.ToTensor(),
        ]
    )

    train_set_x = FacadesDataset(
        root="data/facades/train", transforms=transforms, category=0
    )
    train_set_y = FacadesDataset(
        root="data/facades/train", transforms=transforms, category=1
    )

    train_loader_x = torch.utils.data.DataLoader(
        dataset=train_set_x,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    train_loader_y = torch.utils.data.DataLoader(
        dataset=train_set_y,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    fixed_inputs_x, _ = next(iter(train_loader_x))
    fixed_inputs_x = fixed_inputs_x.to(device).float()
    fixed_inputs_y, _ = next(iter(train_loader_y))
    fixed_inputs_y = fixed_inputs_y.to(device).float()

    optimizerDX = torch.optim.Adam(netDX.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizerDY = torch.optim.Adam(netDY.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizerF = torch.optim.Adam(netF.parameters(), lr=lr, betas=(0.5, 0.9))

    if os.path.exists("checkpoints/cycle-optim-dx.pytorch"):
        optimizerDX.load_state_dict(torch.load("checkpoints/cycle-optim-dx.pytorch"))
    if os.path.exists("checkpoints/cycle-optim-dy.pytorch"):
        optimizerDY.load_state_dict(torch.load("checkpoints/cycle-optim-dy.pytorch"))
    if os.path.exists("checkpoints/cycle-optim-g.pytorch"):
        optimizerG.load_state_dict(torch.load("checkpoints/cycle-optim-g.pytorch"))
    if os.path.exists("checkpoints/cycle-optim-f.pytorch"):
        optimizerF.load_state_dict(torch.load("checkpoints/cycle-optim-f.pytorch"))

    Lambda = 10  # according to paper authors
    bce_loss = torch.nn.BCELoss()
    mae_loss = torch.nn.L1Loss()

    real_labels = torch.ones(batch_size, 1, 30, 30).to(device)
    fake_labels = torch.zeros(batch_size, 1, 30, 30).to(device)

    num_steps = 1  # len(train_set_x) // batch_size

    iter_x = train_loader_x.__iter__()
    iter_y = train_loader_y.__iter__()

    for epoch in range(num_epochs):
        d_loss, g_loss = 0, 0
        for i in range(num_steps):
            x = next(iter_x)
            y = next(iter_y)
            for k in range(K):
                x = x.to(device).float()
                y = y.to(device).float()
                G_x = netG(x)  # equivalent to getting y back
                F_y = netF(y)  # equivalent to getting x back
                optimizerDX.zero_grad()
                optimizerDY.zero_grad()
                real_outputs = netDX(x)
                fake_outputs = netDX(F_y)
                d_x_1 = bce_loss(real_outputs, real_labels)
                d_g_z_1 = bce_loss(fake_outputs, fake_labels)
                real_outputs = netDY(y)
                fake_outputs = netDY(G_x)
                d_x_2 = bce_loss(real_outputs, real_labels)
                d_g_z_2 = bce_loss(fake_outputs, fake_labels)
                loss = 0.25 * (d_x_1 + d_g_z_1 + d_x_2 + d_g_z_2)
                loss.backward()
                optimizerDX.step()
                optimizerDY.step()
                d_loss += loss
            d_loss /= 2
            x = x.to(device).float()
            y = y.to(device).float()
            G_x = netG(x)  # equivalent to getting y back
            F_y = netF(y)  # equivalent to getting x back
            F_G_x = netF(G_x)  # equivalent to getting x back
            G_F_y = netG(F_y)  # equivalent to getting y back
            netG.zero_grad()
            netF.zero_grad()
            fake_outputs = netDX(F_y)
            d_g_z_1 = bce_loss(fake_outputs, real_labels)
            fake_outputs = netDY(G_x)
            d_g_z_2 = bce_loss(fake_outputs, real_labels)
            loss_a = 0.5 * (mae_loss(F_y, x) + mae_loss(G_x, y))  # Identity Loss
            loss_b = 0.5 * (mae_loss(F_G_x, x) + mae_loss(G_F_y, y))  # Cyclic Loss
            loss = d_g_z_1 + d_g_z_2 + (Lambda * loss_a) + (Lambda * loss_b)  # Tot Loss
            loss.backward()
            optimizerG.step()
            optimizerF.step()
            g_loss += loss
        if epoch % 10 == 0:
            print(
                f"E:{epoch}, G Loss:{g_loss / num_steps}, D Loss:{d_loss / num_steps}"
            )
            torch.save(
                optimizerDX.state_dict(),
                "checkpoints/cyle-optim-dx.pytorch",
            )
            torch.save(
                optimizerDY.state_dict(),
                "checkpoints/cyle-optim-dy.pytorch",
            )
            torch.save(
                optimizerG.state_dict(),
                "checkpoints/cyle-optim-g.pytorch",
            )
            torch.save(
                optimizerF.state_dict(),
                "checkpoints/cyle-optim-f.pytorch",
            )
            torch.save(netDX.state_dict(), "checkpoints/cycle-network-dx.pytorch")
            torch.save(netDY.state_dict(), "checkpoints/cycle-network-dy.pytorch")
            torch.save(netG.state_dict(), "checkpoints/cycle-network-g.pytorch")
            torch.save(netF.state_dict(), "checkpoints/cycle-network-f.pytorch")
            generated_X = netG(fixed_inputs_x).detach().cpu()
            grid = torchvision.utils.make_grid(
                generated_X, nrow=5, padding=10, pad_value=1, normalize=True
            )
            tb_writer.add_image("Cycle/Output", grid, epoch)


@app.command()
def main(
    num_epochs: int = 11,
    batch_size: int = 8,
    lr: float = 2e-4,
    K: int = 1,
    image_width: int = 256,
    image_height: int = 256,
    device: str = "cuda",
):
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    netDX = Discriminator().to(device)
    netDY = Discriminator().to(device)
    netG = Generator().to(device)
    netF = Generator().to(device)

    if os.path.exists("checkpoints/cycle-network-dx.pytorch"):
        netDX.load_state_dict(torch.load("checkpoints/cycle-network-dx.pytorch"))
    if os.path.exists("checkpoints/cycle-network-dy.pytorch"):
        netDY.load_state_dict(torch.load("checkpoints/cycle-network-dy.pytorch"))
    if os.path.exists("checkpoints/cycle-network-g.pytorch"):
        netG.load_state_dict(torch.load("checkpoints/cycle-network-g.pytorch"))
    if os.path.exists("checkpoints/cycle-network-f.pytorch"):
        netF.load_state_dict(torch.load("checkpoints/cycle-network-f.pytorch"))

    train(
        netDX,
        netDY,
        netG,
        netF,
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