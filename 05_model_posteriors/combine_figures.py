# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.pyplot as plt

plots = ["mod1_posteriors.png","mod2_posteriors.png"]

im1,im2 = [Image.open(plots[i]) for i in range(len(plots))]

dst = Image.new('RGB', (im1.width*2 , im1.height), color=(255,255,255))

dst.paste(im1, (0, 0))
dst.paste(im2, (im1.width, 0))

dst.save("posterior_plots.png")

dst.save("BMR-2024-18-04 posterior_plots.pdf")

# dst = Image.new('RGB', (im1.width , im1.height*3), color=(255,255,255))

# dst.paste(im1, (0, 0))
# dst.paste(im2, (0, im1.height))
# dst.paste(im3, (0, im1.height*2))

# dst.save("posterior_plots2.png")




plots = ["mod1_ROC.png","mod2_ROC.png"]

im1,im2 = [Image.open(plots[i]) for i in range(len(plots))]

dst = Image.new('RGB', (im1.width*2 , im1.height), color=(255,255,255))

dst.paste(im1, (0, 0))
dst.paste(im2, (im1.width, 0))

dst.save("roc_plots.png")

dst.save("BMR-2024-18-04 roc_plots.pdf")