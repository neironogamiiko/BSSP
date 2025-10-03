import torch
from torch import nn
from .Layers import *

class BSSPNet(nn.Module):
    """
    Багатошарова сегментаційна мережа.

    Вхід:
        x (torch.Tensor): Вхідний тензор зображення розміром (B, C_in, H, W), де
                                                                                  B — розмір батчу,
                                                                                  C_in — кількість каналів,
                                                                                  H — висота зображення,
                                                                                  W — ширина зображення.

    Вихід:
        tuple(torch.Tensor, torch.Tensor, torch.Tensor): Три вихідні тензори
        (out1, out2, out3) після softmax, з розмірностями, відповідними рівням декодера:
            - out1: найвищий рівень (B, num_classes, H, W)
            - out2: середній рівень (B, num_classes, H/2, W/2)
            - out3: нижчий рівень (B, num_classes, H/4, W/4)
    """
    def __init__(self, in_channels, base=16, scale=2, num_classes=20):
        """
        Ініціалізація BSSPNet.

        :param in_channels: Кількість каналів вхідного зображення.
        :param base: Базова кількість фільтрів.
        :param scale: Множник каналів на кожному рівні MSI.
        :param num_classes: Кількість класів сегментації.
        """
        super(BSSPNet, self).__init__()
        self.base = base
        self.scale = scale
        self.num_classes = num_classes

        # Мультискейлові входи
        self.pool1 = nn.AvgPool2d(2)
        self.pool2 = nn.AvgPool2d(4)
        self.pool3 = nn.AvgPool2d(8)
        self.pool4 = nn.AvgPool2d(16)

        # Енкодер
        self.conv1 = Convolution(in_channels=in_channels,
                                 out_channels=base)
        self.pool_conv1 = nn.AvgPool2d(2)

        self.msi2 = MSI(base, in_channels, base*scale)
        self.conv2 = Convolution(in_channels=base*scale*2,
                                 out_channels=base*scale)
        self.pool_conv2 = nn.AvgPool2d(2)

        self.msi3 = MSI(base*scale, in_channels, base*(scale**2))
        self.conv3 = Convolution(in_channels=base*(scale**2)*2,
                                 out_channels=base*(scale**2))
        self.pool_conv3 = nn.AvgPool2d(2)

        self.msi4 = MSI(base*(scale**2), in_channels, base*(scale**3))
        self.conv4 = Convolution(in_channels=base*(scale**3)*2,
                                 out_channels=base*(scale**3))
        self.pool_conv4 = nn.AvgPool2d(2)

        self.msi5 = MSI(base*(scale**3), in_channels, base*(scale**4))
        self.conv5 = Convolution(in_channels=base*(scale**4)*2,
                                 out_channels=base*(scale**4))

        # Модулі уваги
        self.pam = PAM(base*(scale**4))
        self.cam = CAM()
        self.post_attention_conv = nn.Sequential(
            nn.Conv2d(base*(scale**4),
                      base*(scale**4),
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(base*(scale**4),
                           eps=1e-3),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5)
        )

        # Перший декодер
        self.up6 = UpSample(base*(scale**4),
                            base*(scale**3))
        self.up7 = UpSample(base*(scale**3),
                            base*(scale**2))
        self.up8 = UpSample(base*(scale**2),
                            base*scale)
        self.up9 = UpSample(base*scale,
                            base)
        # Перший вихід
        self.inter_out1 = nn.Conv2d(base, num_classes, kernel_size=3, padding=1)

        # Другий декодер
        self.dec2_conv1 = Convolution(in_channels=base * 2,
                                      out_channels=base)
        self.pool_dec2_1 = nn.AvgPool2d(2)
        self.dec2_conv2 = Convolution(in_channels=base * (1 + scale),
                                      out_channels=base * scale)  # base + base*scale
        self.pool_dec2_2 = nn.AvgPool2d(2)
        self.dec2_conv3 = Convolution(in_channels=base * (scale + (scale ** 2)),
                                      out_channels=base * (scale ** 2))  # base*scale + base*scale^2
        self.pool_dec2_3 = nn.AvgPool2d(2)
        self.dec2_conv4 = Convolution(in_channels=base * ((scale ** 2) + (scale ** 3)),
                                      out_channels=base * (scale ** 3))  # base*scale^2 + base*scale^3
        self.up_dec2_1 = UpSample(in_channels=base * (scale ** 3),
                                  out_channels=base * (scale ** 2))
        self.up_dec2_2 = UpSample(in_channels=base * (scale ** 2),
                                  out_channels=base * scale)
        self.up_dec2_3 = UpSample(in_channels=base * scale,
                                  out_channels=base)
        # Другий вихід
        self.inter_out2 = nn.Conv2d(in_channels=base,
                                    out_channels=num_classes,
                                    kernel_size=3,
                                    padding=1)

        # Третій декодер
        self.dec3_conv1 = Convolution(in_channels=base * 2,
                                      out_channels=base)
        self.pool_dec3_1 = nn.AvgPool2d(2)
        self.dec3_conv2 = Convolution(in_channels=base * (1 + scale),
                                      out_channels=base * scale)
        self.pool_dec3_2 = nn.AvgPool2d(2)
        self.dec3_conv3 = Convolution(in_channels=base * (scale + (scale ** 2)),
                                      out_channels=base * (scale ** 2))
        self.pool_dec3_3 = nn.AvgPool2d(2)
        self.dec3_conv4 = Convolution(in_channels=base * ((scale ** 2) + (scale ** 3)),
                                      out_channels=base * (scale ** 3))
        self.up_dec3_1 = UpSample(in_channels=base * (scale ** 3),
                                  out_channels=base * (scale ** 2))
        self.up_dec3_2 = UpSample(in_channels=base * (scale ** 2),
                                  out_channels=base * scale)
        self.up_dec3_3 = UpSample(in_channels=base * scale,
                                  out_channels=base)
        # Третій вихід
        self.inter_out3 = nn.Conv2d(in_channels=base,
                                    out_channels=num_classes,
                                    kernel_size=3, padding=1)

        # inter1/out1 - кінцевий вихід з верхнього декодера.
        # inter2/out2 - проміжний вихід після up7/up8, тобто середній рівень декодера.
        # inter3/out3 - проміжний вихід після up6/up7, тобто нижній рівень декодера.

        self._init_weights()

    def _init_weights(self):
        """
        Ініціалізація вагів мережі.

        Для Conv2d і ConvTranspose2d використовується kaiming_normal_.
        Для BatchNorm2d ваги = 1, bias = 0.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward-процес:
            1. Обчислюються мультискейлові входи через AvgPool2d.
            2. Проходження через енкодер (MSI + Convolution + pooling).
            3. Обчислення уваги PAM і CAM на останньому шарі.
            4. Об'єднання результатів уваги та обробка post_attention_conv.
            5. Декодер з UpSample-блоками та skip-з'єднаннями.
            6. Обчислення трьох проміжних виходів (out1, out2, out3) з softmax.

        :param x: Вхідний тензор розміру (B, C_in, H, W)
        :return: tuple(torch.Tensor, torch.Tensor, torch.Tensor) - три вихідні тензори
                 з softmax для сегментації на різних рівнях декодера.
        """
        # Мультискейлові входи
        inputs_1 = self.pool1(x) # (B, C_in, H/2, W/2)
        inputs_2 = self.pool2(x) # (B, C_in, H/4, W/4)
        inputs_3 = self.pool3(x) # (B, C_in, H/8, W/8)
        inputs_4 = self.pool4(x) # (B, C_in, H/16, W/16)

        # Енкодер
        conv1 = self.conv1(x) # (B, base, H, W)
        pool1 = self.pool_conv1(conv1) # (B, base, H/2, W/2)

        conv2 = self.msi2(pool1, inputs_1) # (B, base*scale*2, H/2, W/2)
        conv2 = self.conv2(conv2) # (B, base*scale, H/2, W/2)
        pool2 = self.pool_conv2(conv2) # (B, base*scale, H/4, W/4)

        conv3 = self.msi3(pool2, inputs_2) # (B, base*scale**2*2, H/4, W/4)
        conv3 = self.conv3(conv3) # (B, base*scale**2, H/4, W/4)
        pool3 = self.pool_conv3(conv3) # (B, base*scale**2, H/8, W/8)

        conv4 = self.msi4(pool3, inputs_3) # (B, base*scale**3*2, H/8, W/8)
        conv4 = self.conv4(conv4) # (B, base*scale**3, H/8, W/8)
        pool4 = self.pool_conv4(conv4) # (B, base*scale**3, H/16, W/16)

        conv5 = self.msi5(pool4, inputs_4) # (B, base*scale**4*2, H/16, W/16)
        conv5 = self.conv5(conv5) # (B, base*scale**4, H/16, W/16)

        # Модулі уваги
        pam_out = self.pam(conv5) # (B, base*scale**4, H/16, W/16)
        cam_out = self.cam(conv5) # (B, base*scale**4, H/16, W/16)
        feature_sum = pam_out + cam_out # (B, base*scale**4, H/16, W/16)
        feature_sum = self.post_attention_conv(feature_sum) # (B, base*scale**4, H/16, W/16)

        # Перший декодер
        up6 = self.up6(feature_sum, conv4) # (B, base*scale**3, H/8, W/8)
        up7 = self.up7(up6, conv3) # (B, base*scale**2, H/4, W/4)
        up8 = self.up8(up7, conv2) # (B, base*scale, H/2, W/2)
        up9 = self.up9(up8, conv1) # (B, base, H, W)
        # Перший вихід
        out1 = self.inter_out1(up9)  # (B, num_classes, H, W)      - верхній рівень

        # Другий декодер
        cat2_1 = torch.cat([conv1, up9], dim=1)
        d2_1 = self.dec2_conv1(cat2_1)
        p2_1 = self.pool_dec2_1(d2_1)

        cat2_2 = torch.cat([up8, p2_1], dim=1)
        d2_2 = self.dec2_conv2(cat2_2)
        p2_2 = self.pool_dec2_2(d2_2)

        cat2_3 = torch.cat([up7, p2_2], dim=1)
        d2_3 = self.dec2_conv3(cat2_3)
        p2_3 = self.pool_dec2_3(d2_3)

        cat2_4 = torch.cat([up6, p2_3], dim=1)
        d2_4 = self.dec2_conv4(cat2_4)

        d2_up1 = self.up_dec2_1(d2_4, d2_3)
        d2_up2 = self.up_dec2_2(d2_up1, d2_2)
        d2_up3 = self.up_dec2_3(d2_up2, d2_1)
        # Другий вихід
        out2 = self.inter_out2(d2_up3) # (B, num_classes, H/2, W/2)  - середній рівень

        # Третій декодер
        cat3_1 = torch.cat([up9, d2_up3], dim=1)
        d3_1 = self.dec3_conv1(cat3_1)
        p3_1 = self.pool_dec3_1(d3_1)

        cat3_2 = torch.cat([d2_up2, p3_1], dim=1)
        d3_2 = self.dec3_conv2(cat3_2)
        p3_2 = self.pool_dec3_2(d3_2)

        cat3_3 = torch.cat([d2_up1, p3_2], dim=1)
        d3_3 = self.dec3_conv3(cat3_3)
        p3_3 = self.pool_dec3_3(d3_3)

        cat3_4 = torch.cat([d2_4, p3_3], dim=1)
        d3_4 = self.dec3_conv4(cat3_4)

        d3_up1 = self.up_dec3_1(d3_4, d3_3)
        d3_up2 = self.up_dec3_2(d3_up1, d3_2)
        d3_up3 = self.up_dec3_3(d3_up2, d3_1)
        # Третій вихід
        out3 = self.inter_out3(d3_up3) # (B, num_classes, H/4, W/4)  - низький рівень

        # Додатковий апсемпл до розміру повного зображення
        out1 = nn.functional.interpolate(out1, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        out2 = nn.functional.interpolate(out2, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        out3 = nn.functional.interpolate(out3, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        out1 = nn.functional.softmax(out1, dim=1) # верхній рівень
        out2 = nn.functional.softmax(out2, dim=1) # середній рівень
        out3 = nn.functional.softmax(out3, dim=1) # низький рівень
        # out1, out2 та out3 відповідають повному роздільному зображенню.

        return out1, out2, out3

# Перевірка структури виходів:
# if __name__ == "__main__":
#     model = BSSPNet(in_channels=1)
#     x = torch.randn(1, 1, 800, 640)
#     out1, out2, out3 = model(x)
#     print(out1.shape, out2.shape, out3.shape)