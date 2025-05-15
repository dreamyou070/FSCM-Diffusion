from torch import nn

class CustomStructuredConv(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(CustomStructuredConv, self).__init__()  # ✅ 반드시 호출
        conv_in_padding = (3 - 1) // 2
        self.structured_conv_in = nn.Conv2d(in_channels, out_channels,
                                            kernel_size=3, padding=conv_in_padding)

    def forward(self, x):
        return self.structured_conv_in(x)