import torch
from torch import nn, optim

# Необхідно врахувати:
# TF приймає на вході тензори виду NHWC (batch, height, weight, channels).
# PT приймає на вході тензори виду NCHW (batch, channels. height, weight).

# Наприклад:
# Це впливає на Batch Normalization, бо в TF BatchNormalization(axis=-1) означає, що нормалізація відбувається по останньому каналу.
# В PT nn,BatchNorm2d(num_features) нормалізує по переданому каналу, тобто через зміну форми тензорів зміниться канал для нормалізації.

class PAM(nn.Module):
    """
    Position Attention Module (PAM) для 2D-зображень на PyTorch.

    PAM реалізує механізм позиційної уваги, де кожен піксель на зображенні
    може взаємодіяти з усіма іншими пікселями для виділення глобальних залежностей.

    Вхід:
        x (torch.Tensor): Вхідний тензор розміром (B, C, H, W).

    Вихід:
        torch.Tensor: Вихідний тензор того ж розміру, що й вхідний, з застосованою позиційною увагою.
    """
    def __init__(self, in_channels):
        """
        Атрибути:
            gamma (nn.Parameter): Навчальний скаляр, який масштабує вихід уваги.
            in_channels (int): Кількість каналів у вхідному тензорі.
            mid_channels (int): Кількість каналів для внутрішніх перетворень (in_channels // 8).
            conv4b (nn.Conv2d): 1x1 згортка для формування ключів b.
            conv4c (nn.Conv2d): 1x1 згортка для формування ключів c.
            conv4d (nn.Conv2d): 1x1 згортка для формування значень d.
        :param in_channels: Кількість каналів у вхідному тензорі (channels/C).
        """
        super(PAM, self).__init__()

        # gamma як PT nn.Parameter замінює TF weight.

        self.gamma = nn.Parameter(torch.zeros(1))
        # Параметр gamma масштабує внесок модуля уваги.
        # Початково gamma = 0, щоб мережа могла спершу навчитися базових ознак без уваги.

        self.in_channels = in_channels
        self.mid_channels = self.in_channels//8

        # Структурно in_channels -> filters, тобто mid_channels = filters//8
        # nn.Conv2d, за замовчуванням, використовує `kaiming_uniform_`, а TensorFlow використовує `he_normal`
        # Формально, це різниця у ініціалізації, але функціонально вони працюють однаково.

        self.conv4b = nn.Conv2d(in_channels,
                                self.mid_channels,
                                kernel_size=1,
                                bias=False)

        self.conv4c = nn.Conv2d(in_channels,
                                self.mid_channels,
                                kernel_size=1,
                                bias=False)

        self.conv4d = nn.Conv2d(in_channels,
                                self.mid_channels,
                                kernel_size=1,
                                bias=False)

    def forward(self, x):
        """
        Forward-процес:
            1. Перетворює вхід через три 1x1 згортки: b, c, d.
            2. Перетворює тензори `b` і `c` для батчового матричного множення.
            3. Обчислює матрицю уваги (attention_map) через softmax(b @ c).
            4. Множить attention_map на d, щоб отримати вихід уваги.
            5. Застосовує навчальний скаляр gamma та додає залишкове підключення до x.
        :param x: Вхідний тензор розміром (B, C, H, W), де
                                                            B - розмір батчу,
                                                            C - кількість каналів,
                                                            H - висота зображення,
                                                            W - ширина зображення.
        :return: Вихідний тензор того ж розміру, що й вхідний (B, C, H, W),
        з застосованою позиційною увагою та залишковим підключенням.
        """
        batch, channels, height, weight = x.size()

        # TF reshape та transpose -> PT view та permute
        # vec_b: B, H*W, C//8
        # vec_cT: B, C//8, H*W
        # bcT = vec_B @ vec_cT -> B, H*W, H*W

        b = self.conv4b(x).view(batch, self.mid_channels, -1).permute(0,2,1) # (batch, height*weight, cannels//8)
        c = self.conv4c(x).view(batch, self.mid_channels, -1) # (batch, cannels//8, height*weight)
        d = self.conv4d(x).view(batch, channels, -1).permute(0,2,1) # (batch, height*weight, height*weight)

        # Формально, розміри тензорів однакові, а батч-матричне множення має давати точно такий самий результат.

        # TF Activation('softmax'(bcT) відбувається по останній осі. Значить, для PT остання вісь - це `dim=-1`.
        attention_map = nn.functional.softmax(torch.bmm(b, c), dim=-1)
        output = torch.bmm(attention_map, d).permute(0,2,1).view(batch, channels, height. weight)
        output = self.gamma * output + x

        return output

    # Функціонально, PT відповідає TF, з невеликою технічною відмінністю у ініціалізації, що може повпливати на...
    # ... процес тренування, а саме на швидкість збіжності на початкових епохах.

class CAM(nn.Module):
    """
    Channel Attention Module (CAM) — модуль уваги по каналах.

    Цей модуль обчислює, як сильно кожен канал вхідного тензору взаємодіє з іншими каналами,
    і використовує це для масштабування вихідного тензору.
    Основна ідея: створити матрицю уваги між каналами, яка показує взаємозв'язок каналів
    всередині одного зображення.

    Вхід:
        x: тензор розміру (batch, channels, height, width) — PyTorch формат channels_first.

    Вихід:
        out: тензор того ж розміру, що і вхід, з доданим ефектом уваги по каналах.
    """
    def __init__(self, in_channels):
        """
        Ініціалізація CAM.
        :param in_channels: кількість каналів у вхідному тензорі.
        """
        super(CAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        # Параметр gamma масштабує внесок модуля уваги.
        # Початково gamma = 0, щоб мережа могла спершу навчитися базових ознак без уваги.

    # На кожному етапі треба бути уважним з softmax, щоб не переплутати канали.

    def forward(self, x):
        """
        Forward-процес:
            1. Перетворення в (batch, height*width, channels).
            2. Обчислення матриці уваги.
            3. Застосування матриці уваги до векторизованих каналів.
            4. Відновлення форми тензору до PyTorch формату (batch, channels, height, width).
            5. Масштабування та резидуальна сумісність.
        :param x: вхідний тензор (batch, channels, height, width)
        :return: вихідний тензор того ж розміру.
        """
        batch, channels, height, weight = x.size()
        a = x.permute(0,2,3,1).reshape(batch, height*weight, channels)
        # Для кожного прикладу в батчі і кожного каналу channels описуємо вектор розміром height*weight, який містить всі пікселі цього каналу.
        # Тобто, `a` - це векторизоване представлення вхідного зображення по кожному каналу.

        attention_map = nn.functional.softmax(torch.bmm(a.transpose(1,2), a), dim=-1) # (batch, channels, channels)
        # Описуємо матрицю уваги по каналах для кожного прикладу в батчі.
        # Як сильно кожен канал взаємодіє з іншими каналами, в межах одного зображення.

        output = torch.bmm(a, attention_map) # (batch, height*weight, channels)
        output = output.view(batch, height, weight, channels).permute(0,3,1,2) # назад в (batch, channels, height, weight)
        output = self.gamma * output + x
        return output