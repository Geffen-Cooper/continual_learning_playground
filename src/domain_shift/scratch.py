import torch
import matplotlib.pyplot as plt
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
import torchvision.transforms.functional as TF
model.eval()
import numpy as np
import copy
from matplotlib.widgets import Slider, Button, RadioButtons

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
import time
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
# load image
input_image = Image.open("Basketball2.jpeg")

fig, (ax0,ax1, ax2) = plt.subplots(1, 3,figsize=(15, 5),gridspec_kw={'width_ratios': [1,1.5, 3]})
fig.tight_layout(pad=5.0)
samp = Slider(ax0, 'amnt', 0, 3.0, valinit=1)

# unnormalize the tensor
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
og_input_tensor = preprocess(input_image)
og_input_img = get_og(og_input_tensor)

def update(val):
    global og_input_tensor
    global og_input_img
    input_tensor = copy.deepcopy(og_input_tensor)
    og_img = copy.deepcopy(og_input_img)
    amp = samp.val
    input_tensor = TF.adjust_brightness(input_tensor,amp)
    og_img = TF.adjust_brightness(og_img,amp)

    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        resnet18.to('cuda')

    with torch.no_grad():
        output = resnet18(input_batch)
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
        idx = labels.index("basketball")
    except:
        idx=6
    ax2.clear()
    colors = ['b']*5
    if idx < 5:
        colors[idx] = 'r'
    ax2.bar([0,30,60,90,120],top5_prob.cpu(),tick_label=labels,width=25,color=colors)
    ax2.set_ylim(0,1)
samp.on_changed(update)
plt.show()
    # input_tensor = TF.adjust_brightness(input_tensor,amp)
    # og_img = TF.adjust_brightness(og_img,amp)
    # input_tensor = TF.adjust_contrast(input_tensor,1+0.01*i)
    # og_img = TF.adjust_contrast(og_img,1+0.01*i)
    # input_tensor = TF.adjust_contrast(input_tensor,1-0.01*i)
    # og_img = TF.adjust_contrast(og_img,1-0.01*i)
    # input_tensor += torch.randn((input_tensor.size()))
    # og_img += torch.randn((og_img.size()))