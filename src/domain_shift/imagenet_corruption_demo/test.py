import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("/home/gc28692/Projects/continual_learning/continual_learning_playground/src/domain_shift/imagenet_corruption_demo/imgs/Samoyed.JPEG")

fig,ax_array = plt.subplots(1,4,figsize=(15,4))
ax_array[0].imshow(img.resize((1024,1024)))
ax_array[1].imshow(img.resize((224,224)))
ax_array[2].imshow(img.resize((128,128)))
ax_array[3].imshow(img.resize((64,64)))

plt.show()