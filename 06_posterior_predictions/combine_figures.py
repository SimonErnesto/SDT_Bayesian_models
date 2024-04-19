# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.pyplot as plt

plots = ["mod1_posterior_predictives.png",
         "mod2_posterior_predictives.png"]

im1,im2 = [Image.open(plots[i]) for i in range(len(plots))]

dst = Image.new('RGB', (im1.width*2+500 , im1.height), color=(255,255,255))

dst.paste(im1, (0, 0))
dst.paste(im2, (im1.width+500, 0))

dst.save("prediction_plots.png")

dst.save("BMR-2024-18-04 prediction_plots.pdf")

