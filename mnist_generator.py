import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MnistGenerator(nn.Module):
    def __init__(self):
        super(MnistGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()  # Выходные значения в диапазоне [-1, 1]
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)


class MnistDiscriminator(nn.Module):
    def __init__(self):
        super(MnistDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Вероятность того, что изображение является реальным
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.main(x)


if __name__ == '__main__':
    # Гиперпараметры
    batch_size = 64
    learning_rate = 0.0002
    epochs = 50
    latent_dim = 100  # Размерность вектора случайного шума

    # Загрузка данных
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Инициализация моделей
    generator = MnistGenerator()
    discriminator = MnistDiscriminator()

    # Оптимизаторы
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Функция потерь
    criterion = nn.BCELoss()

    # Тренировка модели
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(train_loader):
            real = torch.ones(imgs.size(0), 1)  # Метки для реальных изображений
            fake = torch.zeros(imgs.size(0), 1)  # Метки для фальшивых изображений

            # Обучение дискриминатора
            optimizer_D.zero_grad()

            real_imgs = imgs
            output_real = discriminator(real_imgs)
            loss_real = criterion(output_real, real)

            z = torch.randn(imgs.size(0), latent_dim)  # Генерация случайного шума
            fake_imgs = generator(z)
            output_fake = discriminator(fake_imgs.detach())
            loss_fake = criterion(output_fake, fake)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # Обучение генератора
            optimizer_G.zero_grad()

            output = discriminator(fake_imgs)
            loss_G = criterion(output, real)

            loss_G.backward()
            optimizer_G.step()

        print(f'Epoch [{epoch + 1}/{epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}')

    torch.save(generator.state_dict(), 'models/generator.pth')
    torch.save(discriminator.state_dict(), 'models/discriminator.pth')
