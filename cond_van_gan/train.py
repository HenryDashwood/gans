import os

import numpy as np
import torch
from fastprogress import progress_bar
from matplotlib import pyplot as plt
from torch import nn
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
            cell.set_title(f"{torch.argmax(labels[row * 5 + col])}")
            cell.axis("off")
    plt.axis("off")
    plt.tight_layout()
    fig.savefig(f"./visuals/{epoch}.jpg")
    plt.close()


def plot_losses(g_loss_ls, d_loss_ls):
    fig = plt.figure(figsize=(16, 10))
    plt.plot(g_loss_ls, label="Generator Loss")
    plt.plot(d_loss_ls, label="Discriminator Loss")
    plt.legend()
    plt.show()


def encodeOneHot(labels, label_dim):
    ret = torch.FloatTensor(labels.shape[0], label_dim)
    ret.zero_()
    ret.scatter_(dim=1, index=labels.view(-1, 1), value=1)
    return ret


def train(
    generator,
    discriminator,
    train_loader,
    batch_size: int,
    num_epochs: int,
    noise_dim: int,
    label_dim: int,
    device,
):
    discriminator_optimiser = torch.optim.SGD(
        discriminator.parameters(), lr=0.001, momentum=0.5
    )
    generator_optimiser = torch.optim.SGD(
        generator.parameters(), lr=0.001, momentum=0.5
    )
    criterion = nn.BCELoss()
    num_steps = len(train_loader) // batch_size

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    test_z = (2 * torch.randn(10, noise_dim) - 1).to(device)
    test_y = encodeOneHot(torch.tensor(np.arange(0, 10)), label_dim).to(device)

    d_loss_ls = []
    g_loss_ls = []
    d_lr_ls = []
    g_lr_ls = []

    for epoch in progress_bar(range(num_epochs)):
        d_counter = 0
        g_counter = 0
        d_loss = 0
        g_loss = 0

        for i, (images, labels) in enumerate(train_loader):
            if i == num_steps:
                break
            for _ in range(4):
                real_images = images.to(device)
                real_conditions = encodeOneHot(labels, label_dim).to(device)
                fake_conditions = encodeOneHot(
                    torch.randint(0, 10, (batch_size,)), label_dim
                ).to(device)
                fake_images = generator(
                    (2 * torch.randn(batch_size, noise_dim) - 1).to(device),
                    fake_conditions,
                )
                discriminator_optimiser.zero_grad()
                real_outputs = discriminator(real_images, real_conditions)
                fake_outputs = discriminator(fake_images, fake_conditions)
                d_x = criterion(real_outputs, real_labels)
                d_g_z = criterion(fake_outputs, fake_labels)
                d_x.backward()
                d_g_z.backward()
                discriminator_optimiser.step()
                d_counter += 1
                d_loss = d_x.item() + d_g_z.item()
            z = (2 * torch.randn(batch_size, noise_dim) - 1).to(device)
            y = encodeOneHot(torch.randint(0, 10, (batch_size,)), label_dim).to(device)
            generator.zero_grad()
            outputs = discriminator(generator(z, y), y)
            loss = criterion(outputs, real_labels)
            loss.backward()
            generator_optimiser.step()
            g_counter += 1
            g_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}")
            print(f"Generator loss: {g_loss / g_counter}")
            print(f"Discriminator loss: {d_loss / d_counter}")
            generated = generator(test_z, test_y).detach().cpu().view(-1, 28, 28)
            visualiseGAN(generated, test_y, epoch)
        g_loss_ls.append(g_loss / g_counter)
        d_loss_ls.append(d_loss / d_counter)
    plot_losses(g_loss_ls, d_loss_ls)


@app.command()
def main(
    batch_size: int = 64,
    num_epochs: int = 512,
    noise_dim: int = 100,
    label_dim: int = 10,
    device: str = "cuda",
):
    train_loader = create_dataloader(
        batch_size=batch_size, train=True, shuffle=True, num_workers=0
    )
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    generator = Generator(noise_dim, label_dim).to(device)
    discriminator = Discriminator(label_dim).to(device)

    train(
        generator,
        discriminator,
        train_loader,
        batch_size,
        num_epochs,
        noise_dim,
        label_dim,
        device,
    )


if __name__ == "__main__":
    app()
