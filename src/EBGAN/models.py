import torch
import torch.nn as nn

class Decoder(nn.Module):
    # def getLayer(self, num_input, num_outout, kernel_size, stride, padding, norm):
    #     tconv = nn.ConvTranspose2d(in_channels=num_input,
    #                                    out_channels=num_outout,
    #                                    kernel_size=kernel_size,
    #                                    stride=stride,
    #                                    padding=padding)
    #     if norm == 'bn':
    #         layer = nn.Sequential(tconv,
    #                       nn.BatchNorm2d(num_outout),
    #                       nn.LeakyReLU(negative_slope=0.2, inplace=True))
    #     elif norm == 'sn':
    #         layer = nn.Sequential(tconv,
    #                               nn.LeakyReLU(negative_slope=0.2, inplace=True))
    #     else:
    #         layer = nn.Sequential(tconv,
    #                               nn.LeakyReLU(negative_slope=0.2, inplace=True))
    #     return layer

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                # nn.init.constant_(m.bias, 0)


    def __init__(self, img_dim, latent_dim):
        super(Decoder, self).__init__()
        self.dims = [256, 128, 128, 64, img_dim]

        self.linear0 = nn.Sequential(nn.Linear(in_features=latent_dim, out_features= self.dims[0] * (4 * 4)),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.deconv0 = nn.Sequential(nn.ConvTranspose2d(in_channels=self.dims[0], out_channels=self.dims[1],
                                                        kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(self.dims[1]),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=self.dims[1], out_channels=self.dims[2],
                                                        kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(self.dims[2]),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels=self.dims[2], out_channels=self.dims[3],
                                                        kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(self.dims[3]),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.deconv3 = nn.ConvTranspose2d(in_channels=self.dims[3], out_channels=self.dims[4], kernel_size=4,
                                          stride=2, padding=1)

        # self.deconv0 = self.getLayer(self.dims[0], self.dims[1], kernel_size=4, stride=2, padding=1)   # 4*4*128
        # self.deconv1 = self.getLayer(self.dims[1], self.dims[2], kernel_size=4, stride=2, padding=1)   # 8*8*128
        # self.deconv2 = self.getLayer(self.[2], self.dims[3], kernel_size=4, stride=2, padding=1)   # 16*16*64
        # self.deconv3 = nn.Sequential(nn.ConvTranspose2d(std_channel*1, image_channel, kernel_size=4, stride=2, padding=1))   # 32*32*3

        self.initialize_weights()

    def forward(self, x):
        x = self.linear0(x)
        x = x.view(-1, self.dims[0], 4, 4)
        x = self.deconv0(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = torch.tanh_(x)
        return x



class Encoder(nn.Module):
    # def getLayer(self, num_input, num_output, kernel_size, stride, padding, norm, is_flatten):
    #     if norm == "sp":
    #         conv = spectral_norm(nn.Conv2d(in_channels=num_input,
    #                                    out_channels=num_output,
    #                                    kernel_size=kernel_size,
    #                                    stride=stride,
    #                                    padding=padding))
    #     else:
    #         conv = nn.Conv2d(in_channels=num_input,
    #                                        out_channels=num_output,
    #                                        kernel_size=kernel_size,
    #                                        stride=stride,
    #                                        padding=padding)
    #
    #     lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    #     flatten = nn.Flatten(start_dim=1)
    #
    #     if is_flatten:
    #         layer = nn.Sequential(conv, lrelu, flatten)
    #     else:
    #         layer = nn.Sequential(conv, lrelu)
    #
    #     return layer
    #
    # def sampling(self, mu, log_var):
    #     std = torch.exp(0.5 * log_var)
    #     eps = torch.randn_like(std)
    #     return eps.mul(std).add_(mu)  # return z sample

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                # nn.init.constant_(m.bias, 0)

    def __init__(self, img_dim, latent_dim):
        super(Encoder, self).__init__()
        self.dims = [64, 128, 128, 256]

        self.conv0 = nn.Sequential(nn.Conv2d(in_channels=img_dim, out_channels=self.dims[0], kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=self.dims[0], out_channels=self.dims[1], kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=self.dims[1], out_channels=self.dims[2], kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=self.dims[2], out_channels=self.dims[3], kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.linear0 = nn.Sequential(nn.Flatten(1),
                                     nn.Linear(in_features=self.dims[3] * (4 * 4), out_features=latent_dim),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                     )

        # self.image_size = image_size // 2 ** 4
        # self.channels = [std_channel, std_channel*2, std_channel*4]
        #
        # self.layer1 = self.getLayer(image_channel,    self.channels[0], kernel_size=4, stride=2, padding=1, is_flatten=False, norm=norm)
        # self.layer2 = self.getLayer(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1, is_flatten=False, norm=norm)
        # self.layer3 = self.getLayer(self.channels[1], self.channels[1], kernel_size=4, stride=2, padding=1, is_flatten=False, norm=norm)
        # self.feature = self.getLayer(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1, is_flatten=True, norm=norm)

        # if norm == "sp":
        #     self.fc_lrelu = nn.Sequential(spectral_norm(nn.Linear(self.image_size * self.image_size * self.channels[-1], latent_dim)),
        #                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # else:
        #     self.fc_lrelu = nn.Sequential(nn.Linear(self.image_size * self.image_size * self.channels[-1], latent_dim),
        #                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.linear0(x)
        return x

    def getFeatures(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Embedding_labeled_latent(nn.Module):
    def __init__(self, latent_dim, num_class):
        super(Embedding_labeled_latent, self).__init__()

        self.embedding = nn.Sequential(nn.Embedding(num_embeddings=num_class, embedding_dim=latent_dim),
                                      nn.Flatten())

    def forward(self, z, label):
        le = z * self.embedding(label)
        return le
