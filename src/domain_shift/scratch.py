import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np
import copy
from matplotlib.widgets import Slider, Button, RadioButtons
import cv2
from PIL import Image
from torchvision import transforms
import time
import torchvision.models as models

model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights='MobileNet_V2_Weights.DEFAULT')
model.eval()
resnet18 = models.resnet18(weights='DEFAULT')
resnet18.eval()

fig, (ax0,ax1, ax2) = plt.subplots(1, 3,figsize=(15, 5),gridspec_kw={'width_ratios': [0.5,1.5, 3]})
samp = Slider(ax0, 'amnt', 0, 3.0, valinit=1,orientation="vertical")
axr = fig.add_axes([0.05, 0.7, 0.08, 0.15])
axi = fig.add_axes([0.05, 0.5, 0.08, 0.15])
radio = RadioButtons(axr, ('brightness', 'noise', 'sharpness'), active=0)
radio2 = RadioButtons(axi, ('cup', 'sock', 'water bottle','backpack','live'), active=0)

cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")

# unnormalize the tensor to visualize
def get_og(input_tensor):
    input_tensor = copy.deepcopy(input_tensor).cpu()
    input_tensor[:][:][0]*=(.229)
    input_tensor[:][:][0]+=(0.485)
    input_tensor[:][:][1]*=(.224)
    input_tensor[:][:][1]+=(0.455)
    input_tensor[:][:][2]*=(.225)
    input_tensor[:][:][2]+=(0.405)
    input_tensor = (input_tensor-torch.min(input_tensor))/torch.max(input_tensor-torch.min(input_tensor))
    return input_tensor

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

print("start")
input_image = Image.open("cup.JPEG")
og_input_tensor = preprocess(input_image)
og_input_img = get_og(og_input_tensor)
corr = "brightness"
img_class = "cup"


def update(val):
    global og_input_tensor
    global og_input_img
    global corr
    global img_class

    input_tensor = copy.deepcopy(og_input_tensor)
    og_img = copy.deepcopy(og_input_img)
    amp = samp.val
    if corr == "brightness":
        # print(input_tensor[0][0][0])
        input_tensor = TF.adjust_brightness(input_tensor,amp)
        # print(input_tensor[0][0][0])
        og_img = TF.adjust_brightness(og_img,amp)
    elif corr == "noise":
        input_tensor += ((amp-1))*torch.randn((input_tensor.size()))
        og_img += ((amp-1))*torch.randn(og_img.size())
    elif corr == "sharpness":
        input_tensor = TF.adjust_sharpness(input_tensor,amp)
        og_img = TF.adjust_sharpness(og_img,amp)

    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        # resnet18.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        # output = resnet18(input_batch)
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    ax1.imshow(og_img.permute(1, 2, 0))
    
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    labels = []
    for j in range(top5_prob.size(0)):
        labels.append(categories[top5_catid[j].cpu()])
    try:
        idx = labels.index(img_class)
    except:
        idx=6
    ax2.clear()
    colors = ['b']*5
    if idx < 5:
        colors[idx] = 'r'
    ax2.bar([0,30,60,90,120],top5_prob.cpu(),tick_label=labels,width=25,color=colors)
    ax2.set_ylim(0,1)
    plt.draw()
samp.on_changed(update)

def update2(val):
    global corr
    corr = val
    update(1)
radio.on_clicked(update2)

def update3(val):
    global img_class
    global og_input_tensor
    global og_input_img
    img_class = val
    # load image
    if img_class == "live":
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(frame)
    else:
        input_image = Image.open(img_class+".JPEG")
    og_input_tensor = preprocess(input_image)
    og_input_img = get_og(og_input_tensor)
    update(1)
radio2.on_clicked(update3)

update(1)
plt.show()