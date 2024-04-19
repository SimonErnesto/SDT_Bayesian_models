# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.pyplot as plt

plots = ["mod1_rank_plots.png","mod2_rank_plots.png"]

im1,im2 = [Image.open(plots[i]) for i in range(len(plots))]

dst = Image.new('RGB', (im1.width*2+100, im2.height), color=(255,255,255))

dst.paste(im1, (0, 0))
dst.paste(im2, (im2.width, 0))

dst.save("rank_plots.png")

dst.save("BMR-2024-18-04 SDT-Models-rank_plots.pdf")

# plots = ["mod1_rank_plots.pdf","mod2_rank_plots.pdf","mod3_rank_plots.pdf"]

# im1,im2,im3 = [Image.open(plots[i]) for i in range(len(plots))]

# dst = Image.new('RGB', (im1.width*2 , im1.height*2), color=(255,255,255))

# dst.paste(im1, (0, 0))
# dst.paste(im2, (0, im2.height))
# dst.paste(im3, (im1.width, 0))

# dst.save("BMR-2022-09-06 SDT-Models-rank_plots.pdf")

