import os

import torch
import torch.nn.functional as F
from fastprogress import progress_bar
from torchvision import datasets, transforms
from torchvision.utils import save_image
from typer import Typer

app = Typer()

from fid_score import fid_score
from inception_score import inception_score
from model import Generator, Discriminator


class FlattenTransform:
    def __call__(self, inputs):
        return inputs.view(inputs.shape[0], -1)


def create_dataloader(
    batch_size: int, train: bool, shuffle: bool, num_workers: int = 0
):
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


def train(
    generator,
    discriminator,
    train_loader,
    batch_size: int,
    num_epochs: int,
    noise_dim: int,
    device,
):
    generator_optimiser = torch.optim.SGD(
        generator.parameters(), lr=0.005, momentum=0.7
    )
    discriminator_optimiser = torch.optim.SGD(
        discriminator.parameters(), lr=0.005, momentum=0.7
    )
    criterion = torch.nn.BCELoss()

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    test_set = torch.randn(16, noise_dim).to(device)
    num_steps = len(train_loader) // batch_size

    for epoch in progress_bar(range(num_epochs)):
        for i, (images, _) in enumerate(train_loader):
            if i >= num_steps:
                break
            for _ in range(16):
                real_images = images.to(device)
                fake_images = generator(torch.randn(batch_size, noise_dim).to(device))
                discriminator_optimiser.zero_grad()
                real_outputs = discriminator(real_images)
                fake_outputs = discriminator(fake_images)
                d_x = criterion(real_outputs, real_labels)
                d_g_z = criterion(fake_outputs, fake_labels)
                d_x.backward()
                d_g_z.backward()
                discriminator_optimiser.step()

            z = torch.randn(batch_size, noise_dim).to(device)
            generator.zero_grad()
            outputs = discriminator(generator(z))
            loss = criterion(outputs, real_labels)
            loss.backward()
            generator_optimiser.step()

        if epoch % 10 == 0:
            preds = generator(test_set)
            fakes = preds.detach().cpu().view(-1, 1, 28, 28)
            resized_fakes = F.interpolate(fakes.repeat(1, 3, 1, 1), size=(299, 299))
            fakes_inception_score = inception_score(resized_fakes, batch_size)

            reals, _ = next(iter(train_loader))
            reals = reals.to(device).view(-1, 1, 28, 28)
            resized_reals = F.interpolate(reals.repeat(1, 3, 1, 1), size=(299, 299))
            reals_inception_score = inception_score(resized_reals, batch_size)
            fid_score_result = fid_score(
                resized_reals, resized_fakes, batch_size, device
            )

            _ = save_image(
                fakes,
                f"visuals/{str(epoch)}.jpg",
                nrow=4,
                padding=10,
                pad_value=1,
            )

            print(f"Generated Images Inception Score: {fakes_inception_score}")
            print(f"Real Images Inception Score: {reals_inception_score}")
            print(f"FID Score: {fid_score_result}")


@app.command()
def main(
    batch_size: int = 64,
    num_epochs: int = 512,
    noise_dim: int = 100,
    device: str = "cuda",
):
    train_loader = create_dataloader(
        batch_size=batch_size, train=True, shuffle=True, num_workers=0
    )

    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists("./visuals"):
        os.mkdir("./visuals")

    generator = Generator(noise_dim).to(device)
    discriminator = Discriminator().to(device)

    train(
        generator,
        discriminator,
        train_loader,
        batch_size,
        num_epochs,
        noise_dim,
        device,
    )


if __name__ == "__main__":
    app()