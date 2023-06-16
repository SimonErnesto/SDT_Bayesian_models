# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

cdf = stats.norm.cdf
inv_cdf = stats.norm.ppf
pdf = stats.norm.pdf
       
##plotting paramters
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.titlesize': 16})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.markerfacecolor'] = 'w'
plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.markeredgewidth'] = 2
 
np.random.seed(33)


g = 2 #number of groups (conditions)
p = 100 #number of participants


# simulate experiment where sensitivity (d') is correlated with bias (c)
# as d' increases c decreases  
rho_high = -0.05 #correlation for high sensitivity condition
d_std = 0.5 #d' standard deviation
c_std = 0.5 #c standard deviation
mean = [2, 0.1] #d' mean (2) and c mean (0.5), i.e. high sensitivity and low bias
cov = [[d_std**2, rho_high * d_std * c_std],
       [rho_high * d_std * c_std, c_std**2]] #covariance with correlation 
d_high, c_high = np.random.multivariate_normal(mean, cov, size=p).T #generate correlated variables via an mv normal
correlation_high = np.corrcoef(d_high, c_high)[0, 1]


rho_low = -0.6
d_std = 0.5
c_std = 0.5
mean = [1, 0.5]
cov = [[d_std**2, rho_low * d_std * c_std],
       [rho_low * d_std * c_std, c_std**2]]
d_low, c_low = np.random.multivariate_normal(mean, cov, size=p).T
correlation_low = np.corrcoef(d_low, c_low)[0, 1]


sig = np.array([np.repeat(25, p), np.repeat(25, p)]) #fixed number of signal trials (25) 
noi = np.array([np.repeat(75, p), np.repeat(75, p)]) #fixed number of noise trials (75)

d_prime = np.array([d_high, d_low])
c_bias = np.array([c_high, c_low])

hits = np.random.binomial(sig, cdf(0.5*d_prime - c_bias)) #derive hits from d' and c
fas = np.random.binomial(noi, cdf(-0.5*d_prime - c_bias)) #derive false alarms from d' and c

print("Correlation coefficient low sensitivity:", correlation_high)
print("Correlation coefficient high sensitivity:", correlation_low)


gs = ["high", "low"]
hs = [hits[0].mean().round(2), hits[1].mean().round(2)]
fs = [fas[0].mean().round(2), fas[1].mean().round(2)]
ds = [d_high.mean().round(2), d_low.mean().round(2)]
cs = [c_high.mean().round(2), c_low.mean().round(2)]
corrs = [correlation_high.round(2), correlation_low.round(2)]

table1 = pd.DataFrame({"Group":gs, "Hits":hs, "False Alarms":fs, "d'":ds, "c":cs, "rho":corrs})
table1.to_csv("table_summary.csv", index=False)

##### plot sims
z0 = np.random.normal(0, 0.5, p)
z = np.linspace(min(z0), max(z0), p)
fp = cdf(z - d_high) 
hp = cdf(z + d_high)
x = np.cumsum(fp)/fp.sum()
y = np.cumsum(hp)/hp.sum()
auc = np.trapz(y,x).round(2)
z2 = np.sort(np.linspace(-4, 4, 100))

fig, axs = plt.subplots(2,2, figsize=(14,8))
axs[0,0].plot(x,y, color="purple", label="ROC: Simulations ('observed')")
axs[0,0].scatter(x=None, y=None, color="w", label="AUC: "+str(auc))
axs[0,0].plot(x,x, color="k", linestyle=":", alpha=0.5)
axs[0,0].legend(loc="lower right")
axs[0,0].spines[['right', 'top']].set_visible(False)
axs[0,0].set_ylabel("Hit Rate")
axs[0,0].set_xlabel("False Alarm Rate")
axs[0,0].set_title("Group 1 (high)")
axs[0,0].grid(alpha=0.3)
axs[1,0].set_ylim(0,0.5)
axs[1,0].plot(z2, pdf(z2, np.sort(hp)), color="mediumblue", label="Signal")
axs[1,0].plot(z2, pdf(z2, np.sort(fp)), color="crimson", linestyle="--", label="Noise")
axs[1,0].axvline(cdf(c_high.mean()), ymin=0.1, ymax=0.8, color='k', linestyle=":")
axs[1,0].text(0.4, 0.01, "Φ(c) = "+str(cdf(c_high.mean()).round(2)))
axs[1,0].text(0.1, 0.42, "Φ(d') = "+str(cdf(d_high.mean()).round(2)))
axs[1,0].hlines(y=0.41, xmin=fp.mean(), xmax=hp.mean(), linewidth=2, color='k')
axs[1,0].set_ylabel("Density")
axs[1,0].legend()
axs[1,0].spines[['right', 'top']].set_visible(False)
axs[1,0].grid(alpha=0.3)

z0 = np.random.normal(0, 0.5, p)
z = np.linspace(min(z0), max(z0), p)
fp = cdf(z - d_low) 
hp = cdf(z + d_low)
x = np.cumsum(fp)/fp.sum()
y = np.cumsum(hp)/hp.sum()
auc = np.trapz(y,x).round(2)
z2 = np.sort(np.linspace(-4, 4, 100))

axs[0,1].plot(x,y, color="purple", label="ROC: Simulations ('observed')")
axs[0,1].scatter(x=None, y=None, color="w", label="AUC: "+str(auc))
axs[0,1].plot(x,x, color="k", linestyle=":", alpha=0.5)
axs[0,1].legend(loc="lower right")
axs[0,1].spines[['right', 'top']].set_visible(False)
#axs[0,1].set_ylabel("Hit Rate")
axs[0,1].set_xlabel("False Alarm Rate")
axs[0,1].set_title("Group 2 (low)")
axs[0,1].grid(alpha=0.3)
axs[1,1].set_ylim(0,0.5)
axs[1,1].plot(z2, pdf(z2, np.sort(hp)), color="mediumblue", label="Signal")
axs[1,1].plot(z2, pdf(z2, np.sort(fp)), color="crimson", linestyle="--", label="Noise")
axs[1,1].axvline(cdf(c_high.mean()), ymin=0.1, ymax=0.8, color='k', linestyle=":")
axs[1,1].text(0.4, 0.01, "Φ(c) = "+str(c_low.mean().round(2)))
axs[1,1].text(0.1, 0.42, "Φ(d') = "+str(cdf(d_low.mean()).round(2)))
axs[1,1].hlines(y=0.41, xmin=fp.mean(), xmax=hp.mean(), linewidth=2, color='k')
axs[1,1].legend()
axs[1,1].spines[['right', 'top']].set_visible(False)
axs[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("simulations_summary.png", dpi=300)
plt.show()
plt.close()