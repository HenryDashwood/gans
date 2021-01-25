import torch
from fastprogress import progress_bar
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
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
    return data_loader


def calc_gradient_penalty(x, y, discriminator, device):
    x = torch.squeeze(x, dim=1)
    alpha = torch.rand(x.shape).to(device)
    interpolated = alpha * x + (1 - alpha) * y
    var_interpolated = torch.autograd.Variable(interpolated, requires_grad=True).to(
        device
    )
    outputs = discriminator(var_interpolated)
    gradients = torch.autograd.grad(
        outputs=outputs,
        inputs=var_interpolated,
        grad_outputs=torch.ones_like(outputs).to(device),
        retain_graph=True,
        create_graph=True,
        only_inputs=False,
        allow_unused=False,
    )[0]
    grad_norm = gradients.norm(2, dim=1).mean()
    loss = (grad_norm - 1) ** 2
    return loss


def train(
    generator,
    discriminator,
    train_loader,
    batch_size: int,
    num_epochs: int,
    noise_dim: int,
    K_steps: int,
    Lambda_value: int,
    device,
):
    tb_writer = SummaryWriter()
    tb_writer.add_text("GRAD_PEN_WGAN", "Init", 0)

    discriminator_optimiser = torch.optim.Adam(
        discriminator.parameters(), lr=0.0001, betas=(0, 0.9)
    )
    generator_optimiser = torch.optim.Adam(
        generator.parameters(), lr=0.0001, betas=(0, 0.9)
    )
    criterion = nn.BCELoss()

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    data_iter = iter(train_loader)

    for epoch in progress_bar(range(num_epochs)):
        # Train Discriminator
        discriminator_loss = 0
        for k in range(K_steps):
            try:
                images, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, _ = next(data_iter)
            real_images = images.to(device)
            fake_images = generator(torch.randn(batch_size, noise_dim).to(device))
            discriminator_optimiser.zero_grad()
            real_outputs = discriminator(real_images)
            fake_outputs = discriminator(fake_images)
            d_x = criterion(real_outputs, real_labels)
            d_g_x = criterion(fake_outputs, fake_labels)
            gan_loss = d_g_x - d_x
            gradient_penalty = calc_gradient_penalty(
                real_images, fake_images, discriminator, device
            )
            loss = gan_loss + Lambda_value * gradient_penalty
            loss.backward()
            discriminator_loss += loss
            # tb_writer.add_scalar(
            #     "discriminator/gradients",
            #     discriminator.fcn[-1].weight.grad.abs().mean().item(),
            #     int(f"{epoch} {k}"),
            # )
            tb_writer.add_graph(discriminator, real_images)
            tb_writer.add_graph(discriminator, fake_images)
            discriminator_optimiser.step()
        discriminator_loss /= K_steps
        tb_writer.add_scalar("discriminator/loss", discriminator_loss.item(), epoch)
        # Train Generator
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, _ = next(data_iter)
        real_images = images.to(device)
        z = torch.randn(batch_size, noise_dim).to(device)
        generator.zero_grad()
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        gan_loss = criterion(outputs, real_labels)
        gradient_penalty = calc_gradient_penalty(
            real_images, fake_images, discriminator, device
        )
        loss = gan_loss + Lambda_value * gradient_penalty
        loss.backward()
        tb_writer.add_scalar("generator/loss", loss.item(), epoch)
        # tb_writer.add_scalar(
        #     "generator/gradients",
        #     generator.fcn[-2].weight.grad.abs().mean().item(),
        #     epoch,
        # )
        tb_writer.add_graph(generator, z)
        generator_optimiser.step()
        if epoch % 10 == 0:
            test_set = torch.randn(25, noise_dim).to(device)
            generated = generator(test_set).detach().cpu().view(-1, 1, 28, 28)
            grid = torchvision.utils.make_grid(
                generated, nrow=5, padding=10, pad_value=1
            )
            tb_writer.add_image("generator/outputs", grid, epoch)


@app.command()
def main(
    batch_size: int = 64,
    num_epochs: int = 1024,
    noise_dim: int = 100,
    K_steps: int = 5,
    Lambda_value: int = 10,
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
        K_steps,
        Lambda_value,
        device,
    )


if __name__ == "__main__":
    app()