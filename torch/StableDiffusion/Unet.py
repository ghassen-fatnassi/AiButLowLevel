import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
torch.manual_seed(50)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, negative_slope):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, negative_slope):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(2 * out_c, out_c, negative_slope)
        self.dropout = nn.Dropout(p=0.4)  
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)  
        x = self.conv(x)
        x = self.dropout(x)  
        return x
        
class Unet(nn.Module):
    def __init__(self, num_classes, in_channels=3, start_filts=64, depth=5, negative_slope=0.00001):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.negative_slope = negative_slope
        self.depth = depth
        """ ResNet50 as Encoder """
        self.encoder = smp.encoders.get_encoder("resnet50", in_channels=in_channels, depth=self.depth, weights="imagenet")
        
        encoder_channels = self.encoder.out_channels
        """ Bottleneck """
        self.bottleneck = ConvBlock(encoder_channels[-1], encoder_channels[-2], self.negative_slope)
        """ Decoders with Dropout """
        self.decoders = nn.ModuleList([
            DecoderBlock(encoder_channels[i] + encoder_channels[i-1], encoder_channels[i-1], self.negative_slope) 
            for i in range(len(encoder_channels)-1, 0, -1)
        ])
        """ Classifier """
        self.outputs = nn.Conv2d(encoder_channels[1], num_classes, kernel_size=1)

    def forward(self, inputs):
        """ Encoder Pass """
        skips = self.encoder(inputs)
        x = skips[-1]  
        """ Bottleneck """
        x = self.bottleneck(x)
        """ Decoder Pass with Skip Connections and Dropout """
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[-(i+2)])  
        outputs = self.outputs(x)
        return outputs
        