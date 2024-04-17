# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.pyplot as plt

plots = ["mod1_precision.png",
         "mod2_precision.png",
         "mod3_precision.png"]

im1,im2,im3 = [Image.open(plots[i]) for i in range(len(plots))]

dst = Image.new('RGB', (im1.width*2 , im1.height*2), color=(255,255,255))

dst.paste(im1, (0, 0))
dst.paste(im2, (im1.width, 0))
dst.paste(im3, (0, im1.height))

dst.save("precision_plots.png")

dst.save("BMR-2022-09-06 precision_plots.pdf")


