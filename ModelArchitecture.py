from pydoc import replace
from pyexpat import features
from unittest.mock import inplace

import torch
from torch import nn, optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

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
    def __init__(self, in_channels: int):
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
        self.mid_channels = max(1, self.in_channels // 8)

        # Структурно in_channels -> filters, тобто mid_channels = filters//8
        # nn.Conv2d, за замовчуванням, використовує `kaiming_uniform_`, а TensorFlow використовує `he_normal`
        # Формально, це різниця у ініціалізації, але функціонально вони працюють однаково.

        self.conv4b = nn.Conv2d(self.in_channels,
                                self.mid_channels,
                                kernel_size=1,
                                bias=False)

        self.conv4c = nn.Conv2d(self.in_channels,
                                self.mid_channels,
                                kernel_size=1,
                                bias=False)

        self.conv4d = nn.Conv2d(self.in_channels,
                                self.in_channels,
                                kernel_size=1,
                                bias=False)

        self._reset_parameters()  # Примусово ініціалізуємо Conv2d як he_normal (відповідник TF)

    def _reset_parameters(self):
        # TF: kernel_initializer='he_normal' → PT: kaiming_normal_
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # kaiming_normal_ - це функція ініціалізації, яка заповнює тензор випадковими числами з нормального розподілу.
                # module.weight - ваги шару, які треба ініціалізувати.
                # 'fan_in' означає, що дисперсія масштабована від кількості вхідних нейронів (кількості вхідних каналів * розмір фільтра).
                # Це дозволяє зберегти стабільну дисперсію на вході активації.
                # nonlinearity='relu' - ініціалізація враховує тип активації після шару.
                # Для ReLU треба трохи інший масштаб, ніж для сигмоїди чи tanh, бо ReLU "обрізає" негативні значення.

    def forward(self, x:torch.Tensor) -> torch.Tensor:
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
        batch, channels, height, width = x.size()

        # TF reshape та transpose -> PT view та permute
        # vec_b: B, H*W, C//8
        # vec_cT: B, C//8, H*W
        # bcT = vec_B @ vec_cT -> B, H*W, H*W

        b = self.conv4b(x).view(batch, self.mid_channels, -1).permute(0, 2, 1) # (batch, height*width, cannels//8)
        c = self.conv4c(x).view(batch, self.mid_channels, -1) # (batch, cannels//8, height*width)
        d = self.conv4d(x).view(batch, channels, -1).permute(0, 2, 1)  # (batch, height*width, cannels)

        # Формально, розміри тензорів однакові, а батч-матричне множення має давати точно такий самий результат.

        # TF Activation('softmax'(bcT) відбувається по останній осі. Значить, для PT остання вісь - це `dim=-1`.
        attention_map = nn.functional.softmax(torch.bmm(b, c), dim=-1) # матриця уваги між пікселями
        # Описуємо матрицю уваги по каналах для кожного прикладу в батчі.
        # Як сильно кожен канал взаємодіє з іншими каналами, в межах одного зображення.

        output = torch.bmm(attention_map, d) # Застосування уваги

        # Відновлення розміру та резидуальне додавання
        output = output.view(batch, height, width, channels).permute(0, 3, 1, 2)
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
    def __init__(self):
        """
        Ініціалізація CAM.
        """
        super(CAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        # Параметр gamma масштабує внесок модуля уваги.
        # Початково gamma = 0, щоб мережа могла спершу навчитися базових ознак без уваги.

    # На кожному етапі треба бути уважним з softmax, щоб не переплутати канали.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        batch, channels, height, width = x.size()
        a = x.permute(0,2,3,1).reshape(batch, height*width, channels)
        # Для кожного прикладу в батчі й кожного каналу channels описуємо вектор розміром height*width, який містить всі пікселі цього каналу.
        # Тобто, `a` - це векторизоване представлення вхідного зображення по кожному каналу.

        attention_map = nn.functional.softmax(torch.bmm(a.transpose(1,2), a), dim=-1) # (batch, channels, channels)
        # Описуємо матрицю уваги по каналах для кожного прикладу в батчі.
        # Як сильно кожен канал взаємодіє з іншими каналами, в межах одного зображення.

        output = torch.bmm(a, attention_map) # (batch, height*width, channels)

        # Відновлення розміру та резидуальне додавання
        output = output.view(batch, height, width, channels).permute(0,3,1,2) # назад в (batch, channels, height, width)
        output = self.gamma * output + x
        return output

class MSI(nn.Module):
    """
    Multi-Scale Input Block (MSI) - це блок, який поєднує два тензори з різних джерел,
    приводячи їх до однакової кількості каналів через 1x1 згортку, а потім об’єднує по каналах.

    Вхід:
        x (torch.Tensor): Тензор розміром (batch_size, in_channels, H, W)
        inputs (torch.Tensor): Тензор розміром (batch_size, inputs_channels, H, W)

    Вихід:
        torch.Tensor: Об’єднаний тензор розміром (batch_size, out_channels*2, H, W)
    """
    def __init__(self, main_in_channels: int, aux_in_channels: int, branch_out_channels: int):
        """
        Ініціалізація MSI-блоку.

        :param main_in_channels: кількість каналів першого входу `x`.
        :param aux_in_channels: кількість каналів другого входу `inputs`.
        :param branch_out_channels: кількість каналів після 1x1 згортки для кожного входу.
        """
        super(MSI, self).__init__()
        self.main_in_channels = main_in_channels
        self.aux_in_channels = aux_in_channels
        self.branch_out_channels = branch_out_channels

        # Згортка для першого блоку.
        self.main_conv = nn.Sequential(
            nn.Conv2d(self.main_in_channels,
                      self.branch_out_channels,
                      kernel_size=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(self.branch_out_channels, eps=1e-3),
            nn.ReLU(inplace=False)
        )

        # Згортка для другого блоку.
        self.aux_conv = nn.Sequential(
            nn.Conv2d(self.aux_in_channels,
                      self.branch_out_channels,
                      kernel_size=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(self.branch_out_channels, eps=1e-3), # Встановлено eps=1e-3 для відповідності TF epsilon
            nn.ReLU(inplace=False)
        )

        self._reset_parameters()  # Примусова ініціалізація ваг як у TF

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, main_input: torch.Tensor, aux_input: torch.Tensor) -> torch.Tensor:
        """
        Forward-процес:
            1. Пропуск першого входу `x` через 1x1 згортку, BatchNorm і ReLU:
               - Вихід: (batch_size, out_channels, H, W)
            2. Пропуск другого входу `inputs` через свій 1x1 блок:
               - Вихід: (batch_size, out_channels, H, W)
            3. Конкатенація обох виходів по каналах (dim=1):
               - Результат: (batch_size, out_channels*2, H, W)

        :param main_input: вхідний тензор першого потоку (batch, in_channels, H, W).
        :param aux_input: вхідний тензор другого потоку (batch, inputs_channels, H, W).
        :return: об’єднаний тензор (batch, out_channels*2, H, W).
        """
        main_input = self.main_conv(main_input)
        aux_input = self.aux_conv(aux_input)
        return torch.cat([main_input, aux_input], dim=1)

    # PyTorch `nn.BatchNorm2d` і TensorFlow `BatchNormalization` обчислюють нормалізацію трохи по-різному.
    # Наприклад, PyTorch використовує eps=1e-5 за замовчуванням, а TF — epsilon=1e-3.
    # Ця різниця у epsilon сильно впливає на точність при маленьких розмірах батчу.

    # Після кількох епох навчання, початкові дрібні числові відмінності нівелюються.
    # Результат моделі на PyTorch та TF буде практично однаковим, якщо архітектура й оптимізація збігаються.
    # Єдина відмінність - це початкові ваги, але вони впливають лише на перші кроки тренування.

    # У разі необхідності максимального наближення результатів можна:
    # 1) виставити eps=1e-3 у PyTorch BatchNorm2d;
    # 2) не використовувати inplace=True для ReLU.

class Convolution(nn.Module):
    def __init__(self, in_channel, out_channel, eps=1e-3, momentum=.99):
        super(Convolution, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channel,
                           eps=eps,
                           momentum=momentum),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=out_channel,
                      out_channels=out_channel,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channel,
                           eps=eps,
                           momentum=momentum),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.conv_block(x)

def _init_he_normal(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if getattr(module, 'bias', None) is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class BSSPNet(nn.Module):
    def __init__(self, in_channels, base=16, scale=2, num_classes=20,
                 eps=1e-3, momentum=.99, dropout=.5):
        super(BSSPNet, self).__init__()
        self.in_channels = in_channels
        self.base = base
        self.scale = scale
        self.num_classes = num_classes
        self.eps = eps
        self.momentum = momentum
        self.dropout = dropout

        self.conv_1 = Convolution(in_channel=in_channels,
                                  out_channel=base,
                                  eps=eps,
                                  momentum=momentum)
        # AvgPool2d in forward

        def branch_out_for(level):
            target = base * (scale ** level)
            return target // 2

        self.msi_2 = MSI(main_in_channels=base,
                        aux_in_channels=in_channels,
                        branch_out_channels=branch_out_for(1))

        self.conv_2 = Convolution(in_channel=base * scale,
                                  out_channel=base * scale,
                                  eps=eps,
                                  momentum=momentum)

        self.msi_3 = MSI(main_in_channels=base * scale,
                        aux_in_channels=in_channels,
                        branch_out_channels=branch_out_for(2))

        self.conv_3 = Convolution(in_channel= base * (scale ** 2),
                                  out_channel= base * (scale ** 2),
                                  eps=eps,
                                  momentum=momentum)

        self.msi_4 = MSI(main_in_channels=base * (scale ** 2),
                         aux_in_channels=in_channels,
                         branch_out_channels=branch_out_for(3))

        self.conv_4 = Convolution(in_channel=base * (scale ** 3),
                                  out_channel=base * (scale ** 3),
                                  eps=eps,
                                  momentum=momentum)

        self.msi_5 = MSI(main_in_channels=base * (scale ** 3),
                        aux_in_channels=in_channels,
                        branch_out_channels=branch_out_for(4))

        self.conv_5 = Convolution(in_channel=base * (scale ** 4),
                                  out_channel=base * (scale ** 4),
                                  eps=eps,
                                  momentum=momentum)

        self.pam = PAM(in_channels=base * (scale ** 4))
        self.cam = CAM()

        self.feature_conv = nn.Conv2d(in_channels=base * (scale ** 4),
                                      out_channels=base * (scale ** 4),
                                      kernel_size=3,
                                      padding=1,
                                      bias=False)
        self.feature_batchnorm = nn.BatchNorm2d(base * (scale ** 4),
                                                eps=eps,
                                                momentum=momentum)
        self.dropout = nn.Dropout(dropout)

        self.up_6_conv_1 = nn.ConvTranspose2d(in_channels=base * (scale ** 4),
                                              out_channels=base * (scale ** 3),
                                              kernel_size=2,
                                              stride=2,
                                              padding=0,
                                              bias=False)
        self.up_6_conv_2 = nn.ConvTranspose2d(in_channels=base * (scale ** 4),
                                              out_channels=base * (scale ** 3),
                                              kernel_size=2,
                                              stride=4,
                                              padding=0,
                                              bias=False)

        self.conv_6 = Convolution(in_channel=base * (scale ** 3) * 2,
                                  out_channel=base * (scale ** 3),
                                  eps=eps,
                                  momentum=momentum)

        self.conv_7 = Convolution(in_channel=base * (scale ** 2) * 2,
                                  out_channel=base * (scale ** 2),
                                  eps=eps,
                                  momentum=momentum)

        self.conv_8 = Convolution(in_channel=base * scale * 2,
                                  out_channel=base * scale,
                                  eps=eps,
                                  momentum=momentum)

        self.conv_9 = Convolution(in_channel=base * 2,
                                  out_channel=base,
                                  eps=eps,
                                  momentum=momentum)

        self.inter_out_1 = nn.Conv2d(in_channels=base,
                                     out_channels=num_classes,
                                     kernel_size=3,
                                     padding=1)

        self.conv_11 = Convolution(in_channel=base * 2,
                                   out_channel=base,
                                   eps=eps,
                                   momentum=momentum)

        self.conv_22 = Convolution(in_channel=base * scale + base,
                                   out_channel=base * scale,
                                   eps=eps,
                                   momentum=momentum)

        self.conv_33 = Convolution(in_channel=base * (scale ** 2) + base * scale,
                                   out_channel=base * (scale ** 2),
                                   eps=eps,
                                   momentum=momentum)

        self.conv_44 = Convolution(in_channel=base * (scale ** 3) * 2,
                                   out_channel=base * (scale ** 3),
                                   eps=eps,
                                   momentum=momentum)

        self.conv77 = Convolution(in_channel=base * (scale ** 2) * 2,
                                  out_channel=base * (scale ** 2),
                                  eps=eps,
                                  momentum=momentum)

        self.conv88 = Convolution(in_channel=base * scale * 2,
                                  out_channel=base * scale,
                                  eps=eps,
                                  momentum=momentum)

        self.conv99 = Convolution(in_channel=base * 2,
                                  out_channel=base,
                                  eps=eps,
                                  momentum=momentum)

        self.inter_out_2 = nn.Conv2d(in_channels=base,
                                     out_channels=num_classes,
                                     kernel_size=3,
                                     padding=1)

        self.conv_111 = Convolution(in_channel=base * 2,
                                    out_channel=base,
                                    eps=eps,
                                    momentum=momentum)

        self.conv_222 = Convolution(in_channel=base * scale + base,
                                    out_channel=base * scale,
                                    eps=eps,
                                    momentum=momentum)

        self.conv_333 = Convolution(in_channel=base * (scale ** 2) * 2 + base * scale,
                                    eps=eps,
                                    momentum=momentum)

        self.conv_888 = Convolution(in_channel=base * scale * 2,
                                    out_channel=base * scale,
                                    eps=eps,
                                    momentum=momentum)

        self.conv_999 = Convolution(in_channel=base * 2,
                                    out_channel=base,
                                    eps=eps,
                                    momentum=momentum)

        self.inter_out3 = nn.Conv2d(in_channels=base,
                                    out_channels=num_classes,
                                    kernel_size=3,
                                    padding=1)

        self.apply(_init_he_normal)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = eps
                module.momentum = momentum

    def forward(self, x):
        pass