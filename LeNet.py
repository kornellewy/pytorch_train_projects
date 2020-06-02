# thank for info https://www.youtube.com/watch?v=fcOW-Zyb5Bo

import torch
import torchvision
import torch.nn as nn

# 1x32x32 input -> (5x5),s=1,p=0->

# In the vanilla convolution each kernel convolves over the whole input volume.
# Example: Your input volume has 3 channels (RGB image).
#  Now you would like to create a ConvLayer for this image. Each kernel in your ConvLayer will use all input channels of the input volume.
#  Let’s assume you would like to use a 3 by 3 kernel. This kernel will have 27 weights and 1 bias, since (W * H * input_Channels = 3 * 3 * 3 = 27 weights).
# The number of output channels is the number of different kernels used in your ConvLayer.
#  If you would like to output 64 channels, your layer will have 64 different 3x3 kernels, each with 27 weights and 1 bias.

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        # whodzace zdjecie jest czerno biale
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), stride=(1,1), padding=(0,0))

        self.linear1 = nn.Linear(120,84)
        self.linear2 = nn.Linear(84,10)

    def forward(self, x):
        # 1 warstwa
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # 2 warstwa
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        # 3 warstwa
        x = self.conv3(x)
        x = self.relu(x) # batchsize x 120 x 1 x 1 
        x = x.reshape(x.shape[0], -1) # zostaje batch size i liczna 120 (n, 120)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        
    
if __name__ == "__main__":
    x = torch.randn(64,1,32,32)
    model = LeNet()
    print(model(x).shape) #64 batch siez i 10 prawdopodobiens na kazda klase
           
        
        
