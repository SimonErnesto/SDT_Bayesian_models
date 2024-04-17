# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.pyplot as plt

plots = ["mod1_posteriors.png","mod2_posteriors.png","mod3_posteriors.png"]

im1,im2,im3 = [Image.open(plots[i]) for i in range(len(plots))]

dst = Image.new('RGB', (im1.width*2 , im1.height*2), color=(255,255,255))

dst.paste(im1, (0, 0))
dst.paste(im2, (im1.width, 0))
dst.paste(im3, (0, im1.height))

dst.save("posterior_plots.png")

dst.save("BMR-2022-09-06 posterior_plots.pdf")

# dst = Image.new('RGB', (im1.width , im1.height*3), color=(255,255,255))

# dst.paste(im1, (0, 0))
# dst.paste(im2, (0, im1.height))
# dst.paste(im3, (0, im1.height*2))

# dst.save("posterior_plots2.png")





plots = ["mod1_ROC.png","mod2_ROC.png","mod3_ROC.png"]

im1,im2,im3 = [Image.open(plots[i]) for i in range(len(plots))]

dst = Image.new('RGB', (im1.width*2 , im1.height*2), color=(255,255,255))

dst.paste(im1, (0, 0))
dst.paste(im2, (im1.width, 0))
dst.paste(im3, (0, im1.height))

dst.save("roc_plots.png")

dst.save("BMR-2022-09-06 roc_plots.pdf")