import six
import sys
sys.path.append('../../')
import collections
import cv2
import time
from models.resnet import rf_lw50
from utils.helpers import prepare_img
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image

def create_visual_anno(anno):
    """"""
    assert np.max(anno) <= 10, "only 10 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [0, 0, 0],
        1: [255, 248, 220],  # cornsilk
        2: [100, 149, 237],  # cornflowerblue
        3: [102, 205, 170],  # mediumAquamarine
        4: [205, 133, 63],  # peru
        5: [160, 32, 240],  # purple
        6: [255, 64, 64],  # brown1
        7: [139, 69, 19],  # Chocolate4
        8: [255,0,0],
        9: [0,255,0],
        10:[0,0,255]
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno


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
img_path = "/home/kong/Downloads/d94be52120f2aa2cfbd7c12f10817b04.jpeg"
with torch.no_grad():
    img = np.array(Image.open(img_path))
    img = cv2.resize(img, (512, 512))
    img_attr = img.copy()
    orig_size = img.shape[:2][::-1]

    img_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float()

    img_inp = img_inp.cuda()

    #     plt.imshow(img)
    start = time.time()
    out, landmark, age, gender, exp, box = segmenter(img_inp)
    segm = out[0].data.cpu().numpy().transpose(1, 2, 0)
    end = time.time()
    segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
    segm = segm.argmax(axis=2).astype(np.uint8)
    print("Infer time :", end - start)

landmark = landmark.detach().cpu().numpy()
landmark = landmark.reshape((-1, 2))
landmark = landmark * [512, 512]
segm_rgb = create_visual_anno(segm)
imgkp = img.copy()
for i in range(landmark.shape[0]):
    cv2.circle(imgkp, (int(landmark[i][0]), int(landmark[i][1])), 2, (255, 0, 0), 4)
image_add = cv2.addWeighted(imgkp, 0.8, segm_rgb, 0.2, 0)

age = F.softmax(age)
age = age.detach().cpu().numpy().flatten()
agelist = list(range(3, 80, 5))
agelist = np.array(agelist)
agelast = int(sum(age * agelist))

gender = gender.detach().cpu().numpy()
gender = np.argmax(gender)
exp = exp.detach().cpu().numpy()
exp = np.argmax(exp)

box = box.detach().cpu().numpy().flatten()
x1 = box[0]
y1 = box[1]
x2 = x1 + box[2]
y2 = y1 + box[3]
x1 = x1 * img.shape[1]
x2 = x2 * img.shape[1]
y1 = y1 * img.shape[0]
y2 = y2 * img.shape[0]
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.rectangle(img_attr, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
cv2.putText(img_attr, "Gender : {}".format(invgendermap[gender]), (10, 50), font, 1, (255, 0, 255), 2)
cv2.putText(img_attr, "Age : {}".format(agelast), (10, 90), font, 1, (255, 0, 255), 2)
cv2.putText(img_attr, "Expression : {}".format(invexpressmap[exp]), (10, 130), font, 1, (255, 0, 255), 2)
result = np.hstack((img, segm_rgb, imgkp, img_attr, image_add))
result = Image.fromarray(result.astype(np.uint8))
result.save("face_seg52aaaqq22qqqqqqqqq4aaaaqa45.jpg")
final = np.array(result)
plt.figure(figsize=(20, 20))
plt.imshow(final)