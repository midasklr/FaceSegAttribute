import sys
sys.path.append('../../')
import collections
import cv2
from models.resnet import rf_lw50
import cv2
import numpy as np
import torch


has_cuda = torch.cuda.is_available()
n_classes = 11

net = rf_lw50(n_classes, imagenet=False, pretrained=False)
cpkt = torch.load("../face/checkpoint.pth.tar")['segmenter']
weights = collections.OrderedDict()
for key in cpkt:
    print(key.split('.',1))
    weights[key.split('.',1)[1]] = cpkt[key]

net.load_state_dict(weights)
net = net.cuda()
net.eval()
dummy_input1 = torch.randn(1, 3, 512, 512)
dummy_input1 = dummy_input1.cuda()
torch.onnx.export(net, dummy_input1, "refinenet.onnx", verbose=True)
