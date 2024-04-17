# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(33)

##plotting paramters
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.titlesize': 16})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.markerfacecolor'] = 'w'
plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.markeredgewidth'] = 2

########## Plot Model 1 ####################
mod1 = pd.read_csv("mod1_precision_analysis.csv")

d_high = mod1.obs_d_high
d_low = mod1.obs_d_low
c_high = mod1.obs_c_high
c_low = mod1.obs_c_low

dhm = mod1.d_high_m.values #high d' group means
dh5 = mod1.d_high_h5.values
dh95 = mod1.d_high_h95.values
dlm = mod1.d_low_m.values #low d' group means
dl5 = mod1.d_low_h5.values
dl95 = mod1.d_low_h95.values

chm = mod1.c_high_m.values #high c group means
ch5 = mod1.c_high_h5.values
ch95 = mod1.c_high_h95.values
clm = mod1.c_low_m.values #low c group means
cl5 = mod1.c_low_h5.values
cl95 = mod1.c_low_h95.values

n = np.arange(len(mod1))

fig, axs = plt.subplots(2,2, figsize=(12,10))
axs[0,0].set_ylim(0, 3)
axs[0,0].scatter(n, d_high, color="purple", marker='d', label="Simulated (observed)")
axs[0,0].plot(n, dhm, color="g", label="Posterior Mean", zorder=0)
axs[0,0].fill_between(n, dh5, dh95, color="g", alpha=0.2, label="90% HDI")
axs[0,0].grid(alpha=0.2)
axs[0,0].legend(loc='lower right')
axs[0,0].set_title("Group 1 (high)")
axs[0,0].set_ylabel("d' (sensitivity)", size=18)
axs[0,0].set_xlabel("Number of Simulated Participants")
axs[0,0].set_xticks(np.arange(10), np.arange(10,110,step=10))
axs[0,0].set_axisbelow(True)

axs[0,1].set_ylim(0, 3)
axs[0,1].scatter(n, d_low, color="purple", marker='d', label="Simulated (observed)")
axs[0,1].plot(n, dlm, color="g", label="Posterior Mean", zorder=0)
axs[0,1].fill_between(n, dl5, dl95, color="g", alpha=0.2, label="90% HDI")
axs[0,1].grid(alpha=0.2)
axs[0,1].legend()
axs[0,1].set_title("Group 2 (low)")
axs[0,1].set_xlabel("Number of Simulated Participants")
axs[0,1].set_xticks(np.arange(10), np.arange(10,110,step=10))
axs[0,1].set_axisbelow(True)

axs[1,0].set_ylim(-1, 1)
axs[1,0].scatter(n, c_high, color="purple", marker='d', label="Simulated (observed)")
axs[1,0].plot(n, chm, color="g", label="Posterior Mean", zorder=0)
axs[1,0].fill_between(n, ch5, ch95, color="g", alpha=0.2, label="90% HDI")
axs[1,0].grid(alpha=0.2)
axs[1,0].legend()
#axs[1,0].set_title("Group 1 (high)")
axs[1,0].set_ylabel("c (bias)", size=18)
axs[1,0].set_xlabel("Number of Simulated Participants")
axs[1,0].set_xticks(np.arange(10), np.arange(10,110,step=10))
axs[1,0].set_axisbelow(True)

axs[1,1].set_ylim(-1, 1)
axs[1,1].scatter(n, c_low, color="purple", marker='d', label="Simulated (observed)")
axs[1,1].plot(n, clm, color="g", label="Posterior Mean", zorder=0)
axs[1,1].fill_between(n, cl5, cl95, color="g", alpha=0.2, label="90% HDI")
axs[1,1].grid(alpha=0.2)
axs[1,1].legend()
#axs[1,1].set_title("Group 2 (low)")
axs[1,1].set_xlabel("Number of Simulated Participants")
axs[1,1].set_xticks(np.arange(10), np.arange(10,110,step=10))
axs[1,1].set_axisbelow(True)

axs[0,0].text(s="A", x=-2.5, y=3.5, size=24)
axs[0,0].text(s="Model 1 (Base Model)", x=-1.5, y=3.5, size=20)

plt.tight_layout()
plt.savefig("mod1_precision.png", dpi=800)
plt.show()
plt.close()


########## Plot Model 2 ####################
mod2 = pd.read_csv("mod2_precision_analysis.csv")

d_high = mod2.obs_d_high
d_low = mod2.obs_d_low
c_high = mod2.obs_c_high
c_low = mod2.obs_c_low

dhm = mod2.d_high_m.values #high d' group means
dh5 = mod2.d_high_h5.values
dh95 = mod2.d_high_h95.values
dlm = mod2.d_low_m.values #low d' group means
dl5 = mod2.d_low_h5.values
dl95 = mod2.d_low_h95.values

chm = mod2.c_high_m.values #high c group means
ch5 = mod2.c_high_h5.values
ch95 = mod2.c_high_h95.values
clm = mod2.c_low_m.values #low c group means
cl5 = mod2.c_low_h5.values
cl95 = mod2.c_low_h95.values

n = np.arange(len(mod2))

fig, axs = plt.subplots(2,2, figsize=(12,10))
axs[0,0].set_ylim(0, 3)
axs[0,0].scatter(n, d_high, color="purple", marker='d', label="Simulated (observed)")
axs[0,0].plot(n, dhm, color="g", label="Posterior Mean", zorder=0)
axs[0,0].fill_between(n, dh5, dh95, color="g", alpha=0.2, label="90% HDI")
axs[0,0].grid(alpha=0.2)
axs[0,0].legend(loc='lower right')
axs[0,0].set_title("Group 1 (high)")
axs[0,0].set_ylabel("d' (sensitivity)", size=18)
axs[0,0].set_xlabel("Number of Simulated Participants")
axs[0,0].set_xticks(np.arange(10), np.arange(10,110,step=10))
axs[0,0].set_axisbelow(True)

axs[0,1].set_ylim(0, 3)
axs[0,1].scatter(n, d_low, color="purple", marker='d', label="Simulated (observed)")
axs[0,1].plot(n, dlm, color="g", label="Posterior Mean", zorder=0)
axs[0,1].fill_between(n, dl5, dl95, color="g", alpha=0.2, label="90% HDI")
axs[0,1].grid(alpha=0.2)
axs[0,1].legend()
axs[0,1].set_title("Group 2 (low)")
axs[0,1].set_xlabel("Number of Simulated Participants")
axs[0,1].set_xticks(np.arange(10), np.arange(10,110,step=10))
axs[0,1].set_axisbelow(True)

axs[1,0].set_ylim(-1, 1)
axs[1,0].scatter(n, c_high, color="purple", marker='d', label="Simulated (observed)")
axs[1,0].plot(n, chm, color="g", label="Posterior Mean", zorder=0)
axs[1,0].fill_between(n, ch5, ch95, color="g", alpha=0.2, label="90% HDI")
axs[1,0].grid(alpha=0.2)
axs[1,0].legend()
#axs[1,0].set_title("Group 1 (high)")
axs[1,0].set_ylabel("c (bias)", size=18)
axs[1,0].set_xlabel("Number of Simulated Participants")
axs[1,0].set_xticks(np.arange(10), np.arange(10,110,step=10))
axs[1,0].set_axisbelow(True)

axs[1,1].set_ylim(-1, 1)
axs[1,1].scatter(n, c_low, color="purple", marker='d', label="Simulated (observed)")
axs[1,1].plot(n, clm, color="g", label="Posterior Mean", zorder=0)
axs[1,1].fill_between(n, cl5, cl95, color="g", alpha=0.2, label="90% HDI")
axs[1,1].grid(alpha=0.2)
axs[1,1].legend()
#axs[1,1].set_title("Group 2 (low)")
axs[1,1].set_xlabel("Number of Simulated Participants")
axs[1,1].set_xticks(np.arange(10), np.arange(10,110,step=10))
axs[1,1].set_axisbelow(True)

axs[0,0].text(s="B", x=-2.5, y=3.5, size=24)
axs[0,0].text(s="Model 2 (Varying Model)", x=-1.5, y=3.5, size=20)

plt.tight_layout()
plt.savefig("mod2_precision.png", dpi=800)
plt.show()
plt.close()


########## Plot Model 3 ####################
mod3 = pd.read_csv("mod3_precision_analysis.csv")

d_high = mod3.obs_d_high
d_low = mod3.obs_d_low
c_high = mod3.obs_c_high
c_low = mod3.obs_c_low

dhm = mod3.d_high_m.values #high d' group means
dh5 = mod3.d_high_h5.values
dh95 = mod3.d_high_h95.values
dlm = mod3.d_low_m.values #low d' group means
dl5 = mod3.d_low_h5.values
dl95 = mod3.d_low_h95.values

chm = mod3.c_high_m.values #high c group means
ch5 = mod3.c_high_h5.values
ch95 = mod3.c_high_h95.values
clm = mod3.c_low_m.values #low c group means
cl5 = mod3.c_low_h5.values
cl95 = mod3.c_low_h95.values

n = np.arange(len(mod3))

fig, axs = plt.subplots(2,2, figsize=(12,10))
axs[0,0].set_ylim(0, 3)
axs[0,0].scatter(n, d_high, color="purple", marker='d', label="Simulated (observed)")
axs[0,0].plot(n, dhm, color="g", label="Posterior Mean", zorder=0)
axs[0,0].fill_between(n, dh5, dh95, color="g", alpha=0.2, label="90% HDI")
axs[0,0].grid(alpha=0.2)
axs[0,0].legend(loc='lower right')
axs[0,0].set_title("Group 1 (high)")
axs[0,0].set_ylabel("d' (sensitivity)", size=18)
axs[0,0].set_xlabel("Number of Simulated Participants")
axs[0,0].set_xticks(np.arange(10), np.arange(10,110,step=10))
axs[0,0].set_axisbelow(True)

axs[0,1].set_ylim(0, 3)
axs[0,1].scatter(n, d_low, color="purple", marker='d', label="Simulated (observed)")
axs[0,1].plot(n, dlm, color="g", label="Posterior Mean", zorder=0)
axs[0,1].fill_between(n, dl5, dl95, color="g", alpha=0.2, label="90% HDI")
axs[0,1].grid(alpha=0.2)
axs[0,1].legend()
axs[0,1].set_title("Group 2 (low)")
axs[0,1].set_xlabel("Number of Simulated Participants")
axs[0,1].set_xticks(np.arange(10), np.arange(10,110,step=10))
axs[0,1].set_axisbelow(True)

axs[1,0].set_ylim(-1, 1)
axs[1,0].scatter(n, c_high, color="purple", marker='d', label="Simulated (observed)")
axs[1,0].plot(n, chm, color="g", label="Posterior Mean", zorder=0)
axs[1,0].fill_between(n, ch5, ch95, color="g", alpha=0.2, label="90% HDI")
axs[1,0].grid(alpha=0.2)
axs[1,0].legend()
#axs[1,0].set_title("Group 1 (high)")
axs[1,0].set_ylabel("c (bias)", size=18)
axs[1,0].set_xlabel("Number of Simulated Participants")
axs[1,0].set_xticks(np.arange(10), np.arange(10,110,step=10))
axs[1,0].set_axisbelow(True)

axs[1,1].set_ylim(-1, 1)
axs[1,1].scatter(n, c_low, color="purple", marker='d', label="Simulated (observed)")
axs[1,1].plot(n, clm, color="g", label="Posterior Mean", zorder=0)
axs[1,1].fill_between(n, cl5, cl95, color="g", alpha=0.2, label="90% HDI")
axs[1,1].grid(alpha=0.2)
axs[1,1].legend()
#axs[1,1].set_title("Group 2 (low)")
axs[1,1].set_xlabel("Number of Simulated Participants")
axs[1,1].set_xticks(np.arange(10), np.arange(10,110,step=10))
axs[1,1].set_axisbelow(True)

axs[0,0].text(s="C", x=-2.5, y=3.5, size=24)
axs[0,0].text(s="Model 3 (LKJ Model)", x=-1.5, y=3.5, size=20)

plt.tight_layout()
plt.savefig("mod3_precision.png", dpi=800)
plt.show()
plt.close()
