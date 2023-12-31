# -*- coding: utf-8 -*-
import os
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
np.random.seed(33)

#####plotting parameters
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.titlesize': 12})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

os.chdir(os.getcwd())

file = open("mod1_reloo.obj",'rb')
mod1_reloo = pickle.load(file)


file = open("mod2_reloo.obj",'rb')
mod2_reloo = pickle.load(file)

file = open("mod3_reloo.obj",'rb')
mod3_reloo = pickle.load(file)


models = {'Base Model':mod1_reloo, 'Varying Model':mod2_reloo, 'LKJ Model':mod3_reloo}

loo = az.compare(models, ic="loo", scale="log")

az.plot_compare(loo, insample_dev=True, plot_kwargs={'color_insample_dev':'crimson', 'color_dse':'steelblue'})
plt.xlabel("ELPD LOO")
plt.title("re-LOO Model Comparison", size=12)
plt.grid(alpha=0.3)
plt.legend(prop={'size': 12})
plt.tight_layout()
plt.savefig('model_comp_loo.png', dpi=300)
plt.show()
plt.close()

loo_df = pd.DataFrame(loo)
loo_df.to_csv("model_comp_loo.csv")