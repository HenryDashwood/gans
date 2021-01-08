import os

import numpy as np
import torch
from fastprogress import progress_bar
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms
from typer import Typer

from model import Generator, Discriminator


app = Typer()


class FlattenTransform:
    def __call__(self, inputs):
        return inputs.view(inputs.shape[0], -1)


def create_dataloader(batch_size, train, shuffle, num_workers=0):
    data = datasets.MNIST(
        "./data/mnist",
        train=train,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), FlattenTransform()]),
    )
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return data_loader


def visualiseGAN(images, labels, epoch):
    if not os.path.exists("./visuals"):
        os.mkdir("./visuals")
    fig, axes = plt.subplots(2, 5, figsize=(20, 18))
    fig.suptitle(f"Epoch {epoch}")
    for row, axe in enumerate(axes):
        for col, cell in enumerate(axe):
            cell.imshow(images[row * 5 + col], cmap="gray")
            cell.axis("off")
    plt.axis("off")
    plt.tight_layout()
    fig.savefig(f"./visuals/{epoch}.jpg")
    plt.close()


def train(
    generator,
    discriminator,
    train_loader,
    batch_size: int,
    num_epochs: int,
    noise_dim: int,
    is_emd: bool,
    device,
):
    tb_writer = SummaryWriter()
    tb_writer.add_text("WGAN", "Init", 0)

    discriminator_optimiser = torch.optim.RMSprop(
        discriminator.parameters(), lr=0.00005, momentum=0
    )
    generator_optimiser = torch.optim.RMSprop(
        generator.parameters(), lr=0.00005, momentum=0
    )
    criterion = nn.KLDivLoss()
    num_steps = len(train_loader) // batch_size

    real_labels = torch.ones(1).to(device)
    fake_labels = (-1 * torch.ones(1)).to(device)

    test_set = torch.randn(25, noise_dim).to(device)

    for epoch in progress_bar(range(num_epochs)):
        for i, (images, _) in enumerate(train_loader):
            if i == num_steps:
                break
            discriminator_loss = 0
            for k in range(5):
                real_images = images.to(device)
                fake_images = generator(torch.randn(batch_size, noise_dim).to(device))
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)
                discriminator_optimiser.zero_grad()
                real_outputs = discriminator(real_images)
                fake_outputs = discriminator(fake_images)
                if is_emd:
                    real_outputs.backward(real_labels)
                    fake_outputs.backward(fake_labels)
                    discriminator_loss += real_outputs - fake_outputs
                else:
                    d_x = criterion(real_outputs, real_labels)
                    d_g_z = criterion(fake_outputs, fake_labels)
                    d_x.backward()
                    d_g_z.backward()
                    discriminator_loss += d_x + d_g_z

                tb_writer.add_scalar(
                    "discriminator/gradients",
                    discriminator.fcn[-1].weight.grad.abs().mean().item(),
                    int(f"{epoch}{k}"),
                )
                tb_writer.add_graph(discriminator, real_images)
                tb_writer.add_graph(discriminator, fake_images)

                discriminator_optimiser.step()

            tb_writer.add_scalar(
                "discriminator/loss", discriminator_loss.item() / 5, epoch
            )

            z = torch.randn(batch_size, noise_dim).to(device)
            generator.zero_grad()
            outputs = discriminator(generator(z))
            if is_emd:
                outputs.backward(real_labels)
                tb_writer.add_scalar("generator/loss", -1 * outputs.item(), epoch)
            else:
                loss = criterion(outputs, real_labels)
                loss.backward()
                tb_writer.add_scalar("generator/loss", loss.item(), epoch)
            tb_writer.add_scalar(
                "generator/gradients",
                generator.fcn[-2].weight.grad.abs().mean().item(),
                epoch,
            )
            tb_writer.add_graph(generator, z)
            generator_optimiser.step()

        if epoch % 10 == 0:
            generated = generator(test_set).detach().cpu().view(-1, 1, 28, 28)
            # visualiseGAN(generated, test_y, epoch)
            grid = torchvision.utils.make_grid(
                generated, nrow=5, padding=10, pad_value=1
            )
            tb_writer.add_image("generator/outputs", grid, epoch)


@app.command()
def main(
    batch_size: int = 64,
    num_epochs: int = 1024,
    noise_dim: int = 100,
    is_emd: bool = True,
    device: str = "cuda",
):
    train_loader = create_dataloader(
        batch_size=batch_size, train=True, shuffle=True, num_workers=0
    )
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    generator = Generator(noise_dim).to(device)
    discriminator = Discriminator().to(device)

    train(
        generator,
        discriminator,
        train_loader,
        batch_size,
        num_epochs,
        noise_dim,
        is_emd,
        device,
    )


if __name__ == "__main__":
    app()