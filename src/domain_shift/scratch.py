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

# load models
mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights='MobileNet_V2_Weights.DEFAULT')
mobilenet.eval()
resnet18 = models.resnet18(weights='DEFAULT')
resnet18.eval()

# init gui
fig, (sev_ax,img_ax,acc_ax) = plt.subplots(1, 3,figsize=(15, 5),gridspec_kw={'width_ratios': [0.5,1.5, 3]})
severity = Slider(sev_ax, 'amnt', 0, 3.0, valinit=1,orientation="vertical") # severity of corruptions
corr_ax = fig.add_axes([0.05, 0.7, 0.08, 0.15]) # corruptions
class_ax = fig.add_axes([0.05, 0.5, 0.08, 0.15]) # image type
corr_radio = RadioButtons(corr_ax, ('brightness', 'noise', 'sharpness'), active=0)
class_radio = RadioButtons(class_ax, ('cup', 'sock', 'water bottle','backpack','live'), active=0)


# init video capture object
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")

# unnormalize the imagenet normalization and normalize to [0,1] to visualize
def get_og(input_tensor,ren):
    # get a copy
    input_tensor = copy.deepcopy(input_tensor).cpu()

    # undo imagenet normalization
    input_tensor[:][:][0]*=(.229)
    input_tensor[:][:][0]+=(0.485)
    input_tensor[:][:][1]*=(.224)
    input_tensor[:][:][1]+=(0.455)
    input_tensor[:][:][2]*=(.225)
    input_tensor[:][:][2]+=(0.405)

    # normalize to valid range [0,1]
    if ren:
        input_tensor = (input_tensor-torch.min(input_tensor))/torch.max(input_tensor-torch.min(input_tensor))
    return input_tensor

# get imagenet classes
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# preprocess imagenet inputs
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# on startup use cup by default
glob_input_image = Image.open("cup.JPEG")
glob_corr = "brightness"
glob_img_class = "cup"


def sev_update(val):
    start = time.time()
    # use global variables
    global glob_input_image
    global glob_corr
    global glob_img_class
    
    amp = severity.val
    inf_tensor = preprocess(glob_input_image)
    
    if glob_corr == "brightness":
        inf_tensor = TF.adjust_brightness(inf_tensor,amp)
        # viz_tensor = TF.adjust_brightness(viz_tensor,amp)
    elif glob_corr == "noise":
        inf_tensor += ((amp-1))*torch.randn(inf_tensor.size())
        # viz_tensor += ((amp-1))*torch.randn(viz_tensor.size())
    elif glob_corr == "sharpness":
        inf_tensor = TF.adjust_sharpness(inf_tensor,amp)
        # viz_tensor = TF.adjust_sharpness(viz_tensor,amp)
    
    inf_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inf_tensor)
    viz_tensor = get_og(inf_tensor,ren=(glob_corr=="noise" or glob_corr=="sharpness"))
    input_batch = inf_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        # resnet18.to('cuda')
        mobilenet.to('cuda')

    with torch.no_grad():
        # output = resnet18(input_batch)
        output = mobilenet(input_batch)
        
    
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    img_ax.imshow(viz_tensor.permute(1, 2, 0))
    
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    labels = []
    for j in range(top5_prob.size(0)):
        labels.append(categories[top5_catid[j].cpu()])
    try:
        idx = labels.index(glob_img_class)
    except:
        idx=6
    acc_ax.clear()
    colors = ['b']*5
    if idx < 5:
        colors[idx] = 'r'
    acc_ax.bar([0,30,60,90,120],top5_prob.cpu(),tick_label=labels,width=25,color=colors)
    acc_ax.set_ylim(0,1)
    plt.draw()
    # print(time.time()-start)


def corr_update(val):
    global glob_corr
    glob_corr = val
    sev_update(val)


def class_update(val):
    global glob_img_class
    global glob_input_image
    glob_img_class = val

    # load image
    if glob_img_class == "live":
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            glob_input_image = Image.fromarray(frame)
    else:
        glob_input_image = Image.open(glob_img_class+".JPEG")

    sev_update(1)


severity.on_changed(sev_update)
corr_radio.on_clicked(corr_update)
class_radio.on_clicked(class_update)

sev_update(1)
plt.show()