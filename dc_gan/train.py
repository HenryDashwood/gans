import os
from math import ceil
from pathlib import Path

from fastprogress import progress_bar
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from typer import Typer

from model import Generator, Discriminator, weights_init

app = Typer()


class SquashTransform:
    def __call__(self, inputs):
        return 2 * inputs - 1


def create_dataloader(data_dir: Path, batch_size: int):
    data = torchvision.datasets.ImageFolder(
        data_dir,
        transform=torchvision.transforms.Compose(
            [
                # torchvision.transforms.Resize((64, 64)),
                torchvision.transforms.ToTensor(),
                SquashTransform(),
            ]
        ),
    )
    dl = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    return dl


def train_discriminator(
    images,
    real_labels,
    fake_labels,
    generator,
    discriminator,
    discriminator_optimiser,
    criterion,
    batch_size,
    nz,
    device,
):
    real_images = images.to(device)
    fake_images = generator(torch.randn(batch_size, nz, 1, 1).to(device))
    discriminator_optimiser.zero_grad()
    real_outputs = discriminator(real_images)
    fake_outputs = discriminator(fake_images)
    d_x = criterion(real_outputs, real_labels)
    d_g_z = criterion(fake_outputs, fake_labels)
    d_x.backward()
    d_g_z.backward()
    discriminator_optimiser.step()
    loss = d_x + d_g_z
    return loss


def train_generator(
    real_labels,
    discriminator,
    generator,
    generator_optimiser,
    criterion,
    batch_size,
    nz,
    device,
):
    z = torch.randn(batch_size, nz, 1, 1).to(device)
    generator.zero_grad()
    outputs = discriminator(generator(z))
    loss = criterion(outputs, real_labels)
    loss.backward()
    generator_optimiser.step()
    return loss


def train(
    train_loader,
    discriminator,
    generator,
    discriminator_optimiser,
    generator_optimiser,
    batch_size,
    nz,
    device,
):
    tb_writer = SummaryWriter()

    criterion = torch.nn.BCELoss()
    fixed_noise = torch.randn(25, nz, 1, 1).to(device)
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    num_steps = ceil(600 / batch_size)
    for epoch in progress_bar(range(1000)):
        d_loss = 0
        g_loss = 0
        for i, (images, _) in enumerate(train_loader):
            if i == num_steps:
                break
            for k in range(4):
                d_loss += train_discriminator(
                    images,
                    real_labels,
                    fake_labels,
                    generator,
                    discriminator,
                    discriminator_optimiser,
                    criterion,
                    batch_size,
                    nz,
                    device,
                )
            g_loss = train_generator(
                real_labels,
                discriminator,
                generator,
                generator_optimiser,
                criterion,
                batch_size,
                nz,
                device,
            )
        if epoch % 10 == 0:
            tb_writer.add_scalar("DCGAN/D Loss", d_loss / num_steps / 4, epoch)
            tb_writer.add_scalar("DCGAN/G Loss", g_loss / num_steps, epoch)
            generated = generator(fixed_noise).detach().cpu().view(-1, 3, 64, 64)
            grid = torchvision.utils.make_grid(
                generated, nrow=5, padding=10, pad_value=1, normalize=True
            )
            tb_writer.add_image("DCGAN/Output", grid, epoch)
            torch.save(generator.state_dict(), "./sec_4_lec_2_netG.pytorch")
            torch.save(discriminator.state_dict(), "./sec_4_lec_2_netD.pytorch")
            torch.save(generator_optimiser.state_dict(), "./sec_4_lec_2_optG.pytorch")
            torch.save(
                discriminator_optimiser.state_dict(), "./sec_4_lec_2_optD.pytorch"
            )


@app.command()
def main(
    data_dir: Path = "data/mini-birds",
    batch_size: int = 128,
    # Number of channels in the training images.
    # For color images this is 3
    nc: int = 3,
    # Size of z latent vector (i.e. size of generator input)
    nz: int = 100,
    # Size of feature maps in generator
    ngf: int = 64,
    # Size of feature maps in discriminator
    ndf: int = 64,
    device: str = "cuda",
):
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    discriminator = Discriminator(nz, ndf, nc).to(device)
    generator = Generator(nz, ngf, nc).to(device)
    discriminator.apply(weights_init)
    generator.apply(weights_init)
    discriminator_optimiser = torch.optim.Adam(
        discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    generator_optimiser = torch.optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )

    train_loader = create_dataloader(data_dir, batch_size=batch_size)

    if os.path.exists("./sec_4_lec_2_netG.pytorch"):
        generator.load_state_dict(torch.load("./sec_4_lec_2_netG.pytorch"))
    if os.path.exists("./sec_4_lec_2_netD.pytorch"):
        discriminator.load_state_dict(torch.load("./sec_4_lec_2_netD.pytorch"))
    if os.path.exists("./sec_4_lec_2_optG.pytorch"):
        generator_optimiser.load_state_dict(torch.load("./sec_4_lec_2_optG.pytorch"))
    if os.path.exists("./sec_4_lec_2_optD.pytorch"):
        discriminator_optimiser.load_state_dict(
            torch.load("./sec_4_lec_2_optD.pytorch")
        )

    train(
        train_loader,
        discriminator,
        generator,
        discriminator_optimiser,
        generator_optimiser,
        batch_size,
        nz,
        device,
    )


if __name__ == "__main__":
    app()