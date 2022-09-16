import torch
from torch import nn

from torchvision import datasets, transforms
from torchvision.utils import save_image

from vision_models_playground.models.generative.adverserial.mnist_discriminator import Discriminator
from vision_models_playground.models.generative.adverserial.mnist_generator import Generator


class GAN(nn.Module):
    def __init__(self, generator: nn.Module, discriminator: nn.Module):
        """
        Initialize the GAN.
        :param generator: the generator network
        :param discriminator: the discriminator network
        """

        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

        beta1 = 0.5

        self.discriminator_optim = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(beta1, 0.999))
        self.generator_optim = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(beta1, 0.999))

        self.loss_func = nn.BCELoss()

        self.noise_dim = self.generator.noise_dim

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the GAN.
        :param x: noise vector
        :return: generated images along with their predictions
        """

        img = self.generator(x)
        pred = self.discriminator(x)
        return img, pred

    def predict(self, x: torch.Tensor):
        """
        Predict the probability of the input being real.
        :param x: images to predict on
        :return: predicted probability
        """

        return self.discriminator.predict(x)

    def generate(self, noise: torch.Tensor):
        """
        Generate a set of samples from the generator, using the given noise
        :param noise: The noise to use to generate the samples
        :return: A list of generated images
        """

        return self.generator(noise)

    def generate_sample(self, num_instances: int = 10):
        """
        Generate a set of samples from the generator.
        :param num_instances: number of samples to generate
        :return: a list of generated samples
        """

        with torch.no_grad():
            z = torch.randn(num_instances, self.noise_dim).cuda()
            x_gen = self.generator(z)
            return x_gen

    def generate_quality_samples(self, num_instances: int = 10, step: int = 10):
        """
        Generate a set of quality samples from the generator, using the discriminator predictions.
        :param num_instances: number of samples to generate
        :param step: number of samples to generate at once, to avoid memory issues
        :return: a list of generated samples
        """

        # Init the results list with the proper shape
        generated_sample = self.generate_sample(1)
        batch_size, *generated_shape = generated_sample.shape
        results = torch.zeros(0, *generated_shape).cuda()

        with torch.no_grad():
            while results.shape[0] < num_instances:
                z = torch.randn(step, self.noise_dim).cuda()
                x_gen = self.generator(z)

                good = self.discriminator(x_gen).squeeze()

                mask = good > 0.5
                x_gen = x_gen[mask]
                results = torch.cat([results, x_gen])

            results = results[:num_instances]
            return results

    def train_step(self, x: torch.Tensor):
        """
        Train the GAN for one step.
        :param x: The real images to train on
        :return: The loss of the GAN
        """

        # zero the gradients
        self.discriminator_optim.zero_grad()
        self.generator_optim.zero_grad()

        # generate a batch of images
        z = torch.randn(x.shape[0], self.noise_dim).cuda()
        x_gen = self.generator(z)

        # train the discriminator
        d_loss = self.discriminator_loss(x, x_gen)
        d_loss.backward()
        self.discriminator_optim.step()

        # train the generator
        x_gen = self.generator(z)
        g_loss = self.generator_loss(x_gen)
        g_loss.backward()
        self.generator_optim.step()

        return d_loss, g_loss

    def discriminator_loss(self, x: torch.Tensor, x_gen: torch.Tensor):
        """
        Compute the discriminator loss.
        :param x: The real images
        :param x_gen: The generated images
        :return: The loss of the discriminator
        """

        # real loss
        y = torch.ones(x.shape[0]).cuda()
        y_pred = self.discriminator(x).squeeze()
        real_loss = self.loss_func(y_pred, y)

        # fake loss
        y = torch.zeros(x_gen.shape[0]).cuda()
        y_pred = self.discriminator(x_gen).squeeze()
        fake_loss = self.loss_func(y_pred, y)

        # total loss
        d_loss = real_loss + fake_loss
        return d_loss

    def generator_loss(self, x_gen: torch.Tensor):
        """
        Compute the generator loss.
        :param x_gen: The generated images
        :return: The loss of the generator
        """

        y = torch.ones(x_gen.shape[0]).cuda()
        y_pred = self.discriminator(x_gen).squeeze()
        g_loss = self.loss_func(y_pred, y)
        return g_loss

    def train_epochs(
            self,
            train_loader: torch.utils.data.DataLoader,
            epochs: int = 10,
            print_every: int = 100,
            save_checkpoint: bool = True,
            generate_images: bool = True
    ):
        """
        Train the GAN for a number of epochs.
        :param train_loader: The data loader for the training data
        :param epochs: The number of epochs to train for
        :param print_every: The number of steps to print the loss every
        :param save_checkpoint: Whether to save the model after each epoch
        :param generate_images: Whether to generate images after each print_every steps inside the epoch
        :return: None
        """

        for epoch in range(epochs):
            for i, x in enumerate(train_loader):
                if isinstance(x, tuple):
                    x = x[0]

                x = x.cuda()
                d_loss, g_loss = self.train_step(x)

                if i % print_every != 0:
                    continue

                print('Epoch: {}, Iteration: {}, D_Loss: {}, G_Loss: {}'.format(epoch, i, d_loss, g_loss))

                if not generate_images:
                    continue

                # generate a image
                with torch.no_grad():
                    z = torch.randn(1, self.noise_dim).cuda()
                    x_gen = self.generator(z)

                    # save the image
                    save_image(x_gen, 'images/{}.png'.format(i))

            # save the model
            if not save_checkpoint:
                continue

            torch.save(self.generator.state_dict(), 'models_checkpoints/generator.pt')
            torch.save(self.discriminator.state_dict(), 'models_checkpoints/discriminator.pt')


def main():
    # create GAN
    generator = Generator()
    discriminator = Discriminator()
    gan = GAN(generator, discriminator)

    # put model on cuda
    gan.cuda()

    # create the data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=64,
        shuffle=True
    )

    # train the GAN
    gan.train_epochs(train_loader, epochs=100, print_every=100)


if __name__ == '__main__':
    main()
