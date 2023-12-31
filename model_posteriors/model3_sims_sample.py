# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as at
import arviz as az

cdf = stats.norm.cdf
inv_cdf = stats.norm.ppf
pdf = stats.norm.pdf
        
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


# cumulative density of standard normal CDF a.k.a Phi
def Phi(x):
    #Cumulative distribution function of standard Gaussian
    return 0.5 + 0.5 * pm.math.erf(x / pm.math.sqrt(2))


# Model with LKJ correlations
with pm.Model() as mod_lkj:
    
    rho_high = pm.LKJCorr("rho_high", n=g, eta=2.0)
    d_high_sd = pm.HalfNormal("d_high_sd", 1)
    c_high_sd = pm.HalfNormal("c_high_sd", 1)
    d_high_mean = pm.Normal("d_high_mean", 0, 1)
    c_high_mean = pm.Normal("c_high_mean", 0, 1)
    high_cov = at.stack([[d_high_sd**2, rho_high[0] * d_high_sd * c_high_sd],
                         [rho_high[0] * d_high_sd * c_high_sd, c_high_sd**2]])
    high_means = at.stack([d_high_mean, c_high_mean])
    high = pm.MvNormal("high", mu=high_means, cov=high_cov, shape=(p,g))
    cov_high = pm.Deterministic("cov_high", high_cov)
    
    rho_low = pm.LKJCorr("rho_low", n=g, eta=2.0)
    d_low_sd = pm.HalfNormal("d_low_sd", 1)
    c_low_sd = pm.HalfNormal("c_low_sd", 1)
    d_low_mean = pm.Normal("d_low_mean", 0, 1)
    c_low_mean = pm.Normal("c_low_mean", 0, 1)
    low_cov = at.stack([[d_low_sd**2, rho_low[0] * d_low_sd * c_low_sd],
                         [rho_low[0] * d_low_sd * c_low_sd, c_low_sd**2]])
    low_means = at.stack([d_low_mean, c_low_mean])
    low = pm.MvNormal("low", mu=low_means, cov=low_cov, shape=(p,g))
    cov_low = pm.Deterministic("cov_low", low_cov)
    
    d = at.stack([high[:,0], low[:,0]])
    c = at.stack([high[:,1], low[:,1]])
    
    H = pm.Deterministic('H', Phi(0.5*d - c)) # hit rate 
    F = pm.Deterministic('F', Phi(-0.5*d - c)) # false alarm rate
    
    yh = pm.Binomial('yh', p=H, n=sig, observed=hits) # sampling for Hits, S is number of signal trials
    yf = pm.Binomial('yf', p=F, n=noi, observed=fas) # sampling for FAs, N is number of noise trials
  
with mod_lkj:
    idata = pm.sample(1000, random_seed=33)
    
pos = idata.stack(sample = ['chain', 'draw']).posterior

d_pos_high = pos['high'][:,0,:].values
d_pos_low = pos['low'][:,0,:].values

c_pos_high = pos['high'][:,1,:].values
c_pos_low = pos['low'][:,1,:].values


# def simi(a,b):
#     def sign(x):
#         x2 = x.copy()
#         x2[x2>0] = 1
#         x2[x2<0] = -1
#         return x2
#     c4 = []
#     for i in range(b.shape[0]):
#         num = np.sum(sign(a*b[:,i]) * (abs(a) + abs(b[:,i])))
#         den = 2*np.sum(np.maximum(abs(a),abs(b[:,i])))
#         c4.append(num/den)
#     return np.array(c4)


def H2(a,b):
    h2 = []
    for i in range(b.shape[1]):
       m1 = a.mean()
       m2 = b[:,i].mean()
       s1 = a.std()
       s2 = b[:,i].std()
       r = np.exp(-0.25*((m1-m2)**2)/(s1**2 + s2**2))
       l = np.sqrt( (2*s1*s2)/(s1**2 + s2**2))
       h2.append(1 - l*r)
    return np.array(h2)

    

simi_d_high = H2(d_high, d_pos_high) 
simi_d_low = H2(d_low, d_pos_low) 

simi_c_high = H2(c_high, c_pos_high) 
simi_c_low = H2(c_low, c_pos_low) 

print("High Sim d' mean: "+str(simi_d_high.mean().round(2))+", SD: "+str(simi_d_high.std().round(2)))
print("Low Sim d' mean: "+str(simi_d_low.mean().round(2))+", SD: "+str(simi_d_low.std().round(2)))

print("High Sim c mean: "+str(simi_c_high.mean().round(2))+", SD: "+str(simi_c_high.std().round(2)))
print("Low Sim c mean: "+str(simi_c_low.mean().round(2))+", SD: "+str(simi_c_low.std().round(2)))

##plotting paramters
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.titlesize': 16})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.markerfacecolor'] = 'w'
plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.markeredgewidth'] = 2

simi_dhm =  simi_d_high.mean().round(2)
simi_dh_hdi =  az.hdi(simi_d_high, hdi_prob=0.9).round(2)
simi_dlm =  simi_d_low.mean().round(2)
simi_dl_hdi =  az.hdi(simi_d_low, hdi_prob=0.9).round(2)
simi_chm =  simi_c_high.mean().round(2)
simi_ch_hdi =  az.hdi(simi_c_high, hdi_prob=0.9).round(2)
simi_clm =  simi_c_low.mean().round(2)
simi_cl_hdi =  az.hdi(simi_c_low, hdi_prob=0.9).round(2)

#### polt d' and c comparisons
fig, axs = plt.subplots(2,2, figsize=(14,12)) 
axs[0,0].set_ylim(-1.5, 4)
axs[0,0].scatter(np.arange(p), d_high, color='purple', marker="d", label="Simulated (observed)")
#axs[0,0].scatter(np.arange(p), d_pos_high.mean(axis=1), color='g', alpha=0.7, label="Predicted Mean")
axs[0,0].errorbar(np.arange(p), d_pos_high.mean(axis=1), yerr=d_pos_high.std(axis=1), fmt='go', alpha=0.5, label="Posterior")
axs[0,0].errorbar(x=None, y=None, color="w", label="H²: "+str(simi_dhm)+" "+str(simi_dh_hdi))
axs[0,0].grid(alpha=0.3)
axs[0,0].legend()
axs[0,0].set_xlabel("Simulated Participants")
axs[0,0].set_ylabel("d' (sensitivity)", size=18)
axs[0,0].set_title("Group 1 (high)")
axs[0,0].spines[['right', 'top']].set_visible(False)
axs[0,1].set_ylim(-1.5, 4)
axs[0,1].scatter(np.arange(p), d_low, color='purple', marker="d", label="Simulated (observed)")
#axs[0,1].scatter(np.arange(p), d_pos_low.mean(axis=1), color='g', alpha=0.7, label="Posterior Mean")
axs[0,1].errorbar(np.arange(p), d_pos_low.mean(axis=1), yerr=d_pos_low.std(axis=1), fmt='go', alpha=0.5, label="Posterior")
axs[0,1].errorbar(x=None, y=None, color="w", label="H²: "+str(simi_dlm)+" "+str(simi_dl_hdi))
axs[0,1].grid(alpha=0.3)
axs[0,1].legend()
axs[0,1].set_xlabel("Simulated Participants")
#axs[0,1].set_ylabel("d' (sensitivity) ")
axs[0,1].set_title("Group 2 (low)")
axs[0,1].spines[['right', 'top']].set_visible(False)
axs[1,0].set_ylim(-1.5, 4)
axs[1,0].scatter(np.arange(p), c_high, color='purple', marker="d", label="Simulated (observed)")
#axs[1,0].scatter(np.arange(p), c_pos_high.mean(axis=1), color='g', alpha=0.7, label="Posterior Mean")
axs[1,0].errorbar(np.arange(p), c_pos_high.mean(axis=1), yerr=c_pos_high.std(axis=1), fmt='go', alpha=0.5, label="Posterior")
axs[1,0].errorbar(x=None, y=None, color="w", label="H²: "+str(simi_chm)+" "+str(simi_ch_hdi))
axs[1,0].grid(alpha=0.3)
axs[1,0].legend()
axs[1,0].set_xlabel("Simulated Participants")
axs[1,0].set_ylabel("c (bias)", size=18)
#axs[1,0].set_title("Group 1 (high)")
axs[1,1].spines[['right', 'top']].set_visible(False)
axs[1,1].set_ylim(-1.5, 4)
axs[1,1].scatter(np.arange(p), c_low, color='purple', marker="d", label="Simulated (observed)")
#axs[1,1].scatter(np.arange(p), c_pos_low.mean(axis=1), color='g', alpha=0.7, label="Posterior Mean")
axs[1,1].errorbar(np.arange(p), c_pos_low.mean(axis=1), yerr=c_pos_low.std(axis=1), fmt='go', alpha=0.5, label="Posterior")
axs[1,1].errorbar(x=None, y=None, color="w", label="H²: "+str(simi_clm)+" "+str(simi_cl_hdi))
axs[1,1].grid(alpha=0.3)
axs[1,1].legend()
axs[1,1].set_xlabel("Simulated Participants")
#axs[1,1].set_ylabel("c (bias) ")
#axs[1,1].set_title("Group 2 (low)")
axs[1,1].spines[['right', 'top']].set_visible(False)
axs[0,0].text(s="C", x=-20, y=5, size=24)
axs[0,0].text(s="Model 3 (LKJ Model)", x=-10, y=5, size=20)
plt.tight_layout()
plt.savefig("mod3_posteriors.png", dpi=300)
plt.show()


##plot ROC
z0 = np.random.normal(0, 0.5, p)
z = np.linspace(min(z0), max(z0), p)
fp = cdf(z - d_high) 
hp = cdf(z + d_high)
xoh = np.cumsum(fp)/fp.sum()
yoh = np.cumsum(hp)/hp.sum()
z0 = np.random.normal(0, 0.5, p)
z = np.linspace(min(z0), max(z0), p)
fp = cdf(z - d_low) 
hp = cdf(z + d_low)
xol = np.cumsum(fp)/fp.sum()
yol = np.cumsum(hp)/hp.sum()
xo = np.array([xoh,xol])
yo = np.array([yoh,yol])
auc_o = np.trapz(yo,xo).round(2)

z0 = np.random.normal(0, 0.5, p)
z = np.linspace(min(z0), max(z0), p)
fp = cdf(z - d_pos_high.T).T 
hp = cdf(z + d_pos_high.T).T
xph = np.cumsum(fp, axis=0)/fp.sum(axis=0)
yph = np.cumsum(hp, axis=0)/hp.sum(axis=0)
z0 = np.random.normal(0, 0.5, p)
z = np.linspace(min(z0), max(z0), p)
fp = cdf(z - d_pos_low.T).T 
hp = cdf(z + d_pos_low.T).T
xpl = np.cumsum(fp,axis=0)/fp.sum(axis=0)
ypl = np.cumsum(hp,axis=0)/hp.sum(axis=0)
xp = np.array([xph,xpl])
yp = np.array([yph,ypl])
xpm = xp.mean(axis=2)
xps = xp.std(axis=2)
ypm = yp.mean(axis=2)
auc_p = np.trapz(yp,xp, axis=1)
auc_p_m = auc_p.mean(axis=1).round(2)
auc_p_sd = auc_p.std(axis=1).round(2)


fig, axs = plt.subplots(2, figsize=(9,9))
axs[0].plot(xo[0], yo[0], color="purple", linestyle=":", label="Observed, AUC: "+str(auc_o[0]))
axs[0].plot(xpm[0], ypm[0], color="g", label="Posterior mean, AUC: "+str(auc_p_m[0])+"±"+str(auc_p_sd[0]))
axs[0].fill_between(xpm[0], ypm[0]-2*xps[0], ypm[0]+2*xps[0], color="g", alpha=0.2, label="Posterior 2SD")
axs[0].plot(xpm[0],xpm[0], color='k', linestyle=":", linewidth=2, alpha=0.5)
axs[0].set_xlabel("False Alarm Rate")
axs[0].set_ylabel("Hit Rate")
axs[0].grid(0.1)
axs[0].legend(loc="lower right")
axs[0].set_title("Group 1 (high)")
axs[0].spines[['right', 'top']].set_visible(False)
axs[1].plot(xo[1], yo[1], color="purple", linestyle=":", label="Observed, AUC: "+str(auc_o[1]))
axs[1].plot(xpm[1], ypm[1], color="g", label="Posterior mean, AUC: "+str(auc_p_m[1])+"±"+str(auc_p_sd[1]))
axs[1].fill_between(xpm[1], ypm[1]-2*xps[1], ypm[1]+2*xps[1], color="g", alpha=0.2, label="Posterior 2SD")
axs[1].plot(xpm[1],xpm[1], color='k', linestyle=":", linewidth=2, alpha=0.5)
axs[1].set_xlabel("False Alarm Rate")
axs[1].set_ylabel("Hit Rate")
axs[1].grid(0.1)
axs[1].legend(loc="lower right")
axs[1].set_title("Group 2 (low)")
axs[1].spines[['right', 'top']].set_visible(False)
axs[0].text(s="C", x=-0.2, y=1.3, size=24)
axs[0].text(s="Model 3 (LKJ Model)", x=-0.1, y=1.3, size=20)
plt.tight_layout()
plt.savefig("mod3_ROC.png", dpi=300)
plt.show()
plt.close()
