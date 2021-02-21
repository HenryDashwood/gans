import os
import math
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from typer import Typer

from model import Generator, Discriminator

app = Typer()


class SquashTransform:
    def __call__(self, inputs):
        return 2 * inputs - 1


def create_dataloader(image_size, batch_size, shuffle, num_workers=0):
    data = datasets.ImageFolder(
        root="./data",
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        ),
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return data_loader


def updateLevel(epoch, step, total_epochs, total_steps, max_levels):
    total_spectrum = total_epochs * total_steps
    current_point = epoch * total_steps + step
    return min(max((current_point / total_spectrum) * max_levels, 4), max_levels)


def downsampleMiniBatch(images, level, max_levels):
    level = min(int(math.ceil(level)), max_levels)
    output_size = 4 * (2 ** (level - 1))
    return nn.functional.adaptive_avg_pool2d(images, output_size=output_size)


def train(
    dataloader,
    generator,
    discriminator,
    optimiser_gen,
    optimiser_disc,
    num_epochs,
    batch_size,
    nz,
    max_levels,
    device,
):
    tb_writer = SummaryWriter()

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    fixed_noise = torch.randn(25, nz, 1, 1).to(device)

    num_steps = len(dataloader) // batch_size
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        d_loss = 0
        g_loss = 0
        for i, (images, _) in enumerate(dataloader):
            if i == num_steps:
                break
            level = updateLevel(epoch, i, num_epochs, num_steps, max_levels) + 0.00009
            for k in range(1):
                images = downsampleMiniBatch(images, level, max_levels)
                real_images = images.to(device)
                fake_images = generator(
                    torch.randn(batch_size, nz, 1, 1).to(device), level
                )
                optimiser_disc.zero_grad()
                real_outputs = discriminator(real_images, level)
                fake_outputs = discriminator(fake_images, level)
                d_x = criterion(real_outputs, real_labels)
                d_g_z = criterion(fake_outputs, fake_labels)
                d_x.backward()
                d_g_z.backward()
                optimiser_disc.step()
                d_loss += d_x + d_g_z
            z = torch.randn(batch_size, nz, 1, 1).to(device)
            generator.zero_grad()
            outputs = discriminator(generator(z, level), level)
            loss = criterion(outputs, real_labels)
            loss.backward()
            optimiser_gen.step()
            g_loss += loss
        if epoch % 10 == 0:
            print(f"E:{epoch}, L:{level}")
            print(f"G Loss:{g_loss / num_steps}, D Loss:{d_loss / num_steps}\n")
            generated = generator(fixed_noise, math.ceil(level)).detach().cpu()
            grid = torchvision.utils.make_grid(
                generated, nrow=5, padding=10, pad_value=1, normalize=True
            )
            tb_writer.add_image("PGGAN/Output", grid, epoch)
            torch.save(optimiser_disc.state_dict(), "checkpoints/optim-d.pytorch")
            torch.save(optimiser_gen.state_dict(), "checkpoints/optim-g.pytorch")
            torch.save(discriminator.state_dict(), "checkpoints/network-d.pytorch")
            torch.save(generator.state_dict(), "checkpoints/network-g.pytorch")


@app.command()
def main(
    max_levels: int = 6,
    image_size: int = 128,
    nc: int = 3,  # For color images this is 3
    nz: int = 150,  # Size of z latent vector (i.e. size of generator input)
    ngf: int = 6,  # Size of feature maps in generator
    ndf: int = 6,  # Size of feature maps in discriminator
    batch_size: int = 48,
    num_epochs: int = 2500,
    shuffle: bool = True,
    num_workers: int = 4,
    device: str = "cuda",
):
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    dataloader = create_dataloader(image_size, batch_size, shuffle, num_workers)

    generator = Generator(
        nz=nz, ngf=ngf, nc=nc, image_size=image_size, max_levels=max_levels
    )
    discriminator = Discriminator(
        ndf=ndf, nc=nc, image_size=image_size, max_levels=max_levels
    )

    optimiser_disc = torch.optim.RMSprop(discriminator.parameters(), lr=0.0001)
    optimiser_gen = torch.optim.RMSprop(generator.parameters(), lr=0.0001)

    if os.path.exists("section-5-optim-d.pytorch"):
        optimiser_disc.load_state_dict(torch.load("checkpoints/optim-d.pytorch"))
    if os.path.exists("checkpoints/optim-g.pytorch"):
        optimiser_gen.load_state_dict(torch.load("checkpoints/optim-g.pytorch"))
    if os.path.exists("checkpoints/network-d.pytorch"):
        discriminator.load_state_dict(torch.load("checkpoints/network-d.pytorch"))
    if os.path.exists("scheckpoints/network-g.pytorch"):
        generator.load_state_dict(torch.load("checkpoints/network-g.pytorch"))

    train(
        dataloader,
        generator,
        discriminator,
        optimiser_gen,
        optimiser_disc,
        num_epochs,
        batch_size,
        nz,
        max_levels,
        device,
    )


if __name__ == "__main__":
    app()