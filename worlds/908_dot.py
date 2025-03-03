from PIL import Image, ImageDraw 
import numpy as np
im = Image.open("t0.jpg")
im=    im.resize((400,400), Image.ANTIALIAS)
draw = ImageDraw.Draw(im)
for i in range(400):
    xy =[np.random.choice(400),np.random.choice(400)]
    draw.point(xy, fill=0)
    # write to stdout
im.save("tt.jpg")