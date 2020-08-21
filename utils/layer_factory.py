"""RefineNet-LightWeight-CRP Block

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

def batchnorm(in_planes):
    "batch norm 2d"
    return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias
    )

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride=stride,
                padding=int(kernel_size / 2.0),
                groups=groups,
                bias=False,
            ),
            batchnorm(out_planes),
            nn.ReLU6(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride=stride,
                padding=int(kernel_size / 2.0),
                groups=groups,
                bias=False,
            ),
            batchnorm(out_planes),
        )


class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(
                self,
                "{}_{}".format(i + 1, "outvar_dimred"),
                conv1x1(
                    in_planes if (i == 0) else out_planes,
                    out_planes,
                    stride=1,
                    bias=False,
                ),
            )
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, "{}_{}".format(i + 1, "outvar_dimred"))(top)
            x = top + x
        return x

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=2)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=2, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class LandmarkHead(nn.Module):
    ## 人脸72点landmark回归 ##
    def __init__(self,inchannels=512):
        super(LandmarkHead,self).__init__()
        self.conv3x3 = nn.Conv2d(inchannels,inchannels,kernel_size=(3,3),stride=2,padding=1)
        self.bn = nn.BatchNorm2d(inchannels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(inchannels,144,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv3x3(x)
        out = self.relu(self.bn(x))
        out = self.avgpool(out)
        out = self.conv1x1(out)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], 144)


class AttributeHead(nn.Module):
    ## 年龄/性别/表情 ##
    def __init__(self, inchannels=512):
        super(AttributeHead, self).__init__()
        self.conv3x3 = nn.Conv2d(inchannels, inchannels, kernel_size=(3, 3), stride=2, padding=1)
        self.bn = nn.BatchNorm2d(inchannels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.age_conv = nn.Conv2d(inchannels, inchannels, kernel_size=(3, 3), stride=2, padding=1)
        self.age_bn = nn.BatchNorm2d(inchannels)
        self.gender_conv = nn.Conv2d(inchannels, inchannels, kernel_size=(3, 3), stride=2, padding=1)
        self.gender_bn = nn.BatchNorm2d(inchannels)
        self.exp_conv = nn.Conv2d(inchannels, inchannels, kernel_size=(3, 3), stride=2, padding=1)
        self.exp_bn = nn.BatchNorm2d(inchannels)
        self.age = nn.Linear(inchannels, 16)
        self.gender = nn.Linear(inchannels, 2)
        self.exp = nn.Linear(inchannels, 3)

    def forward(self, x):
        #         print("attr net input:",x.size())
        out = self.conv3x3(x)
        out = self.relu(self.bn(out))
        age = F.relu(self.age_bn(self.age_conv(out)))
        #         print("attribute net before pooling:",age.size())
        age = F.max_pool2d(age, 2)
        age = age.view(-1, self.num_flat_features(age))
        age = self.age(age)
        gender = F.relu(self.gender_bn(self.gender_conv(out)))
        gender = F.max_pool2d(gender, 2)
        gender = gender.view(-1, self.num_flat_features(gender))
        gender = self.gender(gender)
        exp = F.relu(self.exp_bn(self.exp_conv(out)))
        exp = F.max_pool2d(exp, 2)
        exp = exp.view(-1, self.num_flat_features(exp))
        exp = self.exp(exp)
        return age, gender, exp

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features


class AgeHead(nn.Module):
    ## 年龄/性别/表情 ##
    def __init__(self, inchannels=512):
        super(AgeHead, self).__init__()
        self.conv3x3 = nn.Conv2d(inchannels, inchannels, kernel_size=(3, 3), stride=2, padding=1)
        self.bn = nn.BatchNorm2d(inchannels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.age_conv = nn.Conv2d(inchannels, inchannels, kernel_size=(3, 3), stride=2, padding=1)
        self.age_bn = nn.BatchNorm2d(inchannels)
        self.age = nn.Linear(inchannels, 16)

    def forward(self, x):
        #         print("attr net input:",x.size())
        out = self.conv3x3(x)  # 16x16
        out = self.relu(self.bn(out))
        age = F.relu(self.age_bn(self.age_conv(out)))
        #         print("attribute net before pooling:",age.size())
        age = self.avgpool(age)
        age = age.view(-1, self.num_flat_features(age))
        age = self.age(age)

        return age

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features


class GenderHead(nn.Module):
    ## 年龄/性别/表情 ##
    def __init__(self, inchannels=512):
        super(GenderHead, self).__init__()
        self.conv3x3 = nn.Conv2d(inchannels, inchannels, kernel_size=(3, 3), stride=2, padding=1)
        self.bn = nn.BatchNorm2d(inchannels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.gender_conv = nn.Conv2d(inchannels, inchannels, kernel_size=(3, 3), stride=2, padding=1)
        self.gender_bn = nn.BatchNorm2d(inchannels)
        self.gender = nn.Linear(inchannels, 2)

    def forward(self, x):
        #         print("attr net input:",x.size())
        out = self.conv3x3(x)
        out = self.relu(self.bn(out))

        gender = F.relu(self.gender_bn(self.gender_conv(out)))
        gender = self.avgpool(gender)
        gender = gender.view(-1, self.num_flat_features(gender))
        gender = self.gender(gender)

        return gender

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features


class ExpHead(nn.Module):
    ## 年龄/性别/表情 ##
    def __init__(self, inchannels=512):
        super(ExpHead, self).__init__()
        self.conv3x3 = nn.Conv2d(inchannels, inchannels, kernel_size=(3, 3), stride=2, padding=1)
        self.bn = nn.BatchNorm2d(inchannels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.exp_conv = nn.Conv2d(inchannels, inchannels, kernel_size=(3, 3), stride=2, padding=1)
        self.exp_bn = nn.BatchNorm2d(inchannels)

        self.exp = nn.Linear(inchannels, 3)

    def forward(self, x):
        #         print("attr net input:",x.size())
        out = self.conv3x3(x)
        out = self.relu(self.bn(out))

        exp = F.relu(self.exp_bn(self.exp_conv(out)))
        exp = self.avgpool(exp)
        exp = exp.view(-1, self.num_flat_features(exp))
        exp = self.exp(exp)
        return exp

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

class BoxHead(nn.Module):
    ## 人脸72点landmark回归 ##
    def __init__(self,inchannels=512):
        super(BoxHead,self).__init__()
        self.conv3x3 = nn.Conv2d(inchannels,inchannels,kernel_size=(3,3),stride=2,padding=1)
        self.bn = nn.BatchNorm2d(inchannels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(inchannels,4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv3x3(x)
        out = self.relu(self.bn(x))
        out = self.avgpool(out)
        out = self.conv1x1(out)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], 4)