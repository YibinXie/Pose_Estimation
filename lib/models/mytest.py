import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = self._make_layer(3)
        self.layer2 = self._make_layer(3)

    def _make_layer(self, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv2d(3, 3, 3)
            )

        return nn.ModuleList(layers)

    def forward(self, x):
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)

        return x


# x = np.random.random((1, 3, 256, 256))
# x = torch.Tensor(x)
# model = Net()
# print(model(x))
# print(128//4)

# a = (0,)
# print(len(a))

def f():
    return 1, 2, 3

a, b, c = f()
# print(a)

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F

def compute_RF_numerical(net,img_np):
    '''
    @param net: Pytorch network
    @param img_np: numpy array to use as input to the networks, it must be full of ones and with the correct
    shape.
    '''
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.fill_(1)
            if m.bias:
                m.bias.data.fill_(0)
    net.apply(weights_init)
    img_ = Variable(torch.from_numpy(img_np).float(),requires_grad=True)
    out_cnn=net(img_)
    out_shape=out_cnn.size()
    ndims=len(out_cnn.size())
    grad=torch.zeros(out_cnn.size())
    l_tmp=[]
    for i in range(ndims):
        if i==0 or i ==1:#batch or channel
            l_tmp.append(0)
        else:
            l_tmp.append(out_shape[i]/2)
    print(tuple(l_tmp))
    grad[tuple(l_tmp)]=1
    out_cnn.backward(gradient=grad)
    grad_np=img_.grad[0,0].data.numpy()
    idx_nonzeros=np.where(grad_np!=0)
    RF=[np.max(idx)-np.min(idx)+1 for idx in idx_nonzeros]

    return RF

class Net_Tran(nn.Module):
    def __init__(self):
        super(Net_Tran, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 1, output_padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 8, 3, 2, 1, output_padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.deconv1(out)
        out = self.conv2(out)
        out = self.deconv2(out)

        return out

model = Net_Tran().cuda()

a = torch.Tensor(1, 3, 256, 256).cuda()
# print(model(a).size())

from utils.receptive_field import receptive_field, receptive_field_for_unit
# rec = receptive_field(model, (3, 256, 256))
# print(compute_RF_numerical(model, a))

import platform

if platform.system()=='Windows':
    print('Windows系统')
elif platform.system()=='Linux':
    print('Linux系统')
else:
    print('其他')

import os
this_dir = os.path.dirname(__file__)
print(this_dir)
print(os.getcwd())

