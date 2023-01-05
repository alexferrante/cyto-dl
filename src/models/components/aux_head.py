import torch
from monai.networks.blocks import SubpixelUpsample, Convolution


class ProjectionLayer(torch.nn.Module):
    def __init__(self, dim, pool_size):
        super().__init__()
        self.dim = dim
        self.projection = torch.nn.AvgPool3d(kernel_size=[pool_size, 1, 1])

    def __call__(self, x):
        return self.projection(x).squeeze(self.dim)


class AuxHead(torch.nn.Module):
    def __init__(
        self,
        resolution,
        in_channels,
        out_channels,
        final_act,
        n_hr_convs=1,
        n_lr_convs=1,
        dropout=0.1,
        hr_skip_channels=0,
        spatial_dims=3,
        first_layer=torch.nn.Identity(),
    ):
        super().__init__()
        self.resolution = resolution
        conv_input_channels = in_channels
        modules = [first_layer]
        if resolution == "hr":
            conv_input_channels //= 2 ** spatial_dims
            self.upsample = SubpixelUpsample(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=conv_input_channels,
            )

            for i in range(n_hr_convs):
                # first convolution has to include skip connection from input
                modules.append(
                    Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=conv_input_channels + (hr_skip_channels) * (i == 0),
                        out_channels=conv_input_channels,
                        act=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
                        norm="INSTANCE",
                        dropout=dropout,
                    )
                )
        for i in range(n_lr_convs):
            if i == n_lr_convs - 1:
                act = ""  # empty string is identity activation for monai conv
                out_ch = out_channels
            else:
                act = ("leakyrelu", {"inplace": True, "negative_slope": 0.2})
                out_ch = conv_input_channels
            modules.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=conv_input_channels,
                    out_channels=out_ch,
                    act=act,
                    norm="INSTANCE",
                    dropout=dropout,
                )
            )
        modules.append(final_act)
        self.aux_head = torch.nn.Sequential(*modules)

    def forward(self, x, hr_skip):
        if self.resolution == "hr":
            x_hr = self.upsample(x)
            x = torch.cat((x_hr, hr_skip), dim=1)
        return self.aux_head(x)
