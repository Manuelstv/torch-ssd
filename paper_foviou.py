import torch
from numpy import deg2rad
from foviou import find_foviou
from PIL import Image, ImageDraw, ImageFont
from vis import *


b1 = torch.tensor([[ 40,  50,  35,  55],
        [ 30,  60,  60,  60],
        [ 50, -78,  25,  46],
        [ 30,  75,  30,  60],  # Unchanged for 4th, listed for clarity
        [ 40,  70,  25,  30],
        [ 40,  85,  30,  40]])  # Corrected for 6th case


b2 = torch.tensor([[ 35,  20,  37,  50],
        [ 55,  40,  60,  60],
        [ 30, -75,  26,  45],
        [ 60,  75,  30,  60],  # Corrected for 4th case
        [ 60,  85,  30,  30],
        [ 60,  78,  40,  30]])  # Corrected for 6th case

b1 = deg2rad(b1)
b2 = deg2rad(b2)

'''

    Values from the paper

    For the first example: FoV-IoU = 0.235
    For the second example: FoV-IoU = 0.323
    For the third example: FoV-IoU = 0.617
    For the fourth example: FoV-IoU = 0.589
    For the fifth example: FoV-IoU = 0.259
    For the sixth example: FoV-IoU = 0.538

'''

print(find_foviou(b1, b2))

image = torch.zeros([3,300,300])
target_height, target_width = 600, 300
plot_image(image, b1, target_height, target_width)