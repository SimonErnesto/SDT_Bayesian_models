# -*- coding: utf-8 -*-
import os
import pymc as pm
import pytensor.tensor as at
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd

os.chdir(os.getcwd())

cdf = stats.norm.cdf
inv_cdf = stats.norm.ppf
pdf = stats.norm.pdf
        
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



    

# multilevel Model with varying priors for d and c
with pm.Model() as mod_var:
    
    dl = pm.Normal('dl', 0.0, 1.0)
    dz = pm.Normal('dz', 0.0, 1.0, shape=(g,p)) 
    ds = pm.HalfNormal('ds', 1.0)
    d = pm.Deterministic('d', dl + dz*ds) #discriminability d'
    
    cl = pm.Normal('cl', 0.0, 1.0)
    cz = pm.Normal('cz', 0.0, 1.0, shape=(g,p)) 
    cs = pm.HalfNormal('cs', 1.0)
    c = pm.Deterministic('c', cl + cz*cs) #bias c
    
    H = pm.Deterministic('H', Phi(0.5*d - c)) # hit rate
    F = pm.Deterministic('F', Phi(-0.5*d - c)) # false alarm rate
    
    yh = pm.Binomial('yh', p=H, n=sig, observed=hits) # sampling for Hits, sig is number of signal trials
    yf = pm.Binomial('yf', p=F, n=noi, observed=fas) # sampling for FAs, noi is number of noise trials

with mod_var:
    idata_var = pm.sample(1000, random_seed=33, nuts_sampler='numpyro')
    ppc_var = pm.sample_posterior_predictive(idata_var)

var_summ = az.summary(ppc_var, hdi_prob=0.9)    
var_summ_like = az.summary(ppc_var.posterior_predictive, hdi_prob=0.9)   
var_summ = pd.concat([var_summ, var_summ_like]) 
var_summ.to_csv("mod1_posterior_predictive_summary.csv")   

ppc_var = ppc_var.stack(sample = ['chain', 'draw']).posterior_predictive

###### Plot posterior predictive for varying model (model 2)

h_ppc_high = ppc_var['yh'][0,:,:].values
h_ppc_low = ppc_var['yh'][1,:,:].values

f_ppc_high = ppc_var['yf'][0,:,:].values
f_ppc_low = ppc_var['yf'][1,:,:].values

def SDI(a,b):
    return 2*np.minimum(a,b.T).sum(axis=1)/(a.sum() + b.sum(axis=0))
    
SDI_h_high = SDI(hits[0], h_ppc_high) 
SDI_h_low = SDI(hits[1], h_ppc_low) 

SDI_f_high = SDI(fas[0], f_ppc_high) 
SDI_f_low = SDI(fas[1], f_ppc_low) 

SDI_hhm =  SDI_h_high.mean().round(2)
SDI_hh_hdi =  az.hdi(SDI_h_high, hdi_prob=0.9).round(2)
SDI_hlm =  SDI_h_low.mean().round(2)
SDI_hl_hdi =  az.hdi(SDI_h_low, hdi_prob=0.9).round(2)
SDI_fhm =  SDI_f_high.mean().round(2)
SDI_fh_hdi =  az.hdi(SDI_f_high, hdi_prob=0.9).round(2)
SDI_flm =  SDI_f_low.mean().round(2)
SDI_fl_hdi =  az.hdi(SDI_f_low, hdi_prob=0.9).round(2)


#### plot varying model
fig, axs = plt.subplots(2,2, figsize=(14,12)) 
axs[0,0].set_ylim(-1, 50)
axs[0,0].scatter(np.arange(p), hits[0], color='purple', marker="d", label="Simulated (observed)")
#axs[0,0].scatter(np.arange(p), h_ppc_high.mean(axis=1), color='g', alpha=0.7, label="Predicted Mean")
axs[0,0].errorbar(np.arange(p), h_ppc_high.mean(axis=1), yerr=h_ppc_high.std(axis=1), fmt='go', alpha=0.5, label="Predicted")
axs[0,0].errorbar(x=None, y=None, color="w", label="SDI: "+str(SDI_hhm)+" "+str(SDI_hh_hdi))
axs[0,0].grid(alpha=0.2)
axs[0,0].legend()
axs[0,0].set_xlabel("Simulated Participants")
axs[0,0].set_ylabel("Hits", size=18)
axs[0,0].set_title("Group 1 (high)")
axs[0,0].spines[['right', 'top']].set_visible(False)
axs[0,1].set_ylim(-1, 50)
axs[0,1].scatter(np.arange(p), hits[1], color='purple', marker="d", label="Simulated (observed)")
#axs[0,1].scatter(np.arange(p), h_ppc_low.mean(axis=1), color='g', alpha=0.7, label="Posterior Mean")
axs[0,1].errorbar(np.arange(p), h_ppc_low.mean(axis=1), yerr=h_ppc_low.std(axis=1), fmt='go', alpha=0.5, label="Predicted")
axs[0,1].errorbar(x=None, y=None, color="w", label="SDI: "+str(SDI_hlm)+" "+str(SDI_hl_hdi))
axs[0,1].grid(alpha=0.2)
axs[0,1].legend()
axs[0,1].set_xlabel("Simulated Participants")
#axs[0,1].set_ylabel("d' (sensitivity) ")
axs[0,1].set_title("Group 2 (low)")
axs[0,1].spines[['right', 'top']].set_visible(False)
axs[1,0].set_ylim(-1, 70)
axs[1,0].scatter(np.arange(p), fas[0], color='purple', marker="d", label="Simulated (observed)")
#axs[1,0].scatter(np.arange(p), f_ppc_high.mean(axis=1), color='g', alpha=0.7, label="Posterior Mean")
axs[1,0].errorbar(np.arange(p), f_ppc_high.mean(axis=1), yerr=f_ppc_high.std(axis=1), fmt='go', color='g', alpha=0.5, label="Predicted")
axs[1,0].errorbar(x=None, y=None, color="w", label="SDI: "+str(SDI_fhm)+" "+str(SDI_fh_hdi))
axs[1,0].grid(alpha=0.2)
axs[1,0].legend()
axs[1,0].set_xlabel("Simulated Participants")
axs[1,0].set_ylabel("False Alarms", size=18)
#axs[1,0].set_title("Group 1 (high)")
axs[1,0].spines[['right', 'top']].set_visible(False)
axs[1,1].set_ylim(-1, 70)
axs[1,1].scatter(np.arange(p), fas[1], color='purple', marker="d", label="Simulated (observed)")
#axs[1,1].scatter(np.arange(p), f_ppc_low.mean(axis=1), color='g', alpha=0.7, label="Posterior Mean")
axs[1,1].errorbar(np.arange(p), f_ppc_low.mean(axis=1), yerr=f_ppc_low.std(axis=1), fmt='go', color='g', alpha=0.5, label="Predicted")
axs[1,1].errorbar(x=None, y=None, color="w", label="SDI: "+str(SDI_flm)+" "+str(SDI_fl_hdi))
axs[1,1].grid(alpha=0.2)
axs[1,1].legend()
axs[1,1].set_xlabel("Simulated Participants")
#axs[1,1].set_ylabel("c (bias) ")
#axs[1,1].set_title("Group 2 (low)")
axs[1,1].spines[['right', 'top']].set_visible(False)
axs[0,0].text(s="A", x=-20, y=60, size=24)
axs[0,0].text(s="Model 1 (Varying Model)", x=-10, y=60, size=20)
plt.tight_layout()
plt.savefig("mod1_posterior_predictives.png", dpi=800)
plt.show()





# multilevel Model with LKJ correlations between d and c
with pm.Model() as mod_lkj:
    
    rho_high = pm.LKJCorr("rho_high", n=g, eta=2.0)
    d_high_sd = pm.HalfNormal("d_high_sd", 1.0)
    c_high_sd = pm.HalfNormal("c_high_sd", 1.0)
    d_high_mean = pm.Normal("d_high_mean", 0, 1.0)
    c_high_mean = pm.Normal("c_high_mean", 0, 1.0)
    high_cov = at.stack([[d_high_sd**2, rho_high[0] * d_high_sd * c_high_sd],
                         [rho_high[0] * d_high_sd * c_high_sd, c_high_sd**2]])
    high_means = at.stack([d_high_mean, c_high_mean])
    high = pm.MvNormal("high", mu=high_means, cov=high_cov, shape=(p,g))
    cov_high = pm.Deterministic("cov_high", high_cov)
    
    rho_low = pm.LKJCorr("rho_low", n=g, eta=2.0)
    d_low_sd = pm.HalfNormal("d_low_sd", 1.0)
    c_low_sd = pm.HalfNormal("c_low_sd", 1.0)
    d_low_mean = pm.Normal("d_low_mean", 0, 1.0)
    c_low_mean = pm.Normal("c_low_mean", 0, 1.0)
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
    idata_lkj = pm.sample(1000, random_seed=33, nuts_sampler='numpyro')
    ppc_lkj = pm.sample_posterior_predictive(idata_lkj)

lkj_summ = az.summary(ppc_lkj, hdi_prob=0.9)    
lkj_summ_like = az.summary(ppc_lkj.posterior_predictive, hdi_prob=0.9)   
lkj_summ = pd.concat([lkj_summ, lkj_summ_like]) 
lkj_summ.to_csv("mod2_posterior_predictive_summary.csv")  

ppc_lkj = ppc_lkj.stack(sample = ['chain', 'draw']).posterior_predictive


###### Plot posterior predictive for varying model (model 2)

h_ppc_high = ppc_lkj['yh'][0,:,:].values
h_ppc_low = ppc_lkj['yh'][1,:,:].values

f_ppc_high = ppc_lkj['yf'][0,:,:].values
f_ppc_low = ppc_lkj['yf'][1,:,:].values

def SDI(a,b):
    return 2*np.minimum(a,b.T).sum(axis=1)/(a.sum() + b.sum(axis=0))
    
SDI_h_high = SDI(hits[0], h_ppc_high) 
SDI_h_low = SDI(hits[1], h_ppc_low) 

SDI_f_high = SDI(fas[0], f_ppc_high) 
SDI_f_low = SDI(fas[1], f_ppc_low) 

SDI_hhm =  SDI_h_high.mean().round(2)
SDI_hh_hdi =  az.hdi(SDI_h_high, hdi_prob=0.9).round(2)
SDI_hlm =  SDI_h_low.mean().round(2)
SDI_hl_hdi =  az.hdi(SDI_h_low, hdi_prob=0.9).round(2)
SDI_fhm =  SDI_f_high.mean().round(2)
SDI_fh_hdi =  az.hdi(SDI_f_high, hdi_prob=0.9).round(2)
SDI_flm =  SDI_f_low.mean().round(2)
SDI_fl_hdi =  az.hdi(SDI_f_low, hdi_prob=0.9).round(2)


#### plot LKJ model
#### plot base model
fig, axs = plt.subplots(2,2, figsize=(14,12)) 
axs[0,0].set_ylim(-1, 50)
axs[0,0].scatter(np.arange(p), hits[0], color='purple', marker="d", label="Simulated (observed)")
#axs[0,0].scatter(np.arange(p), h_ppc_high.mean(axis=1), color='g', alpha=0.7, label="Predicted Mean")
axs[0,0].errorbar(np.arange(p), h_ppc_high.mean(axis=1), yerr=h_ppc_high.std(axis=1), fmt='go', alpha=0.5, label="Predicted")
axs[0,0].errorbar(x=None, y=None, color="w", label="SDI: "+str(SDI_hhm)+" "+str(SDI_hh_hdi))
axs[0,0].grid(alpha=0.2)
axs[0,0].legend()
axs[0,0].set_xlabel("Simulated Participants")
axs[0,0].set_ylabel("Hits", size=18)
axs[0,0].set_title("Group 1 (high)")
axs[0,0].spines[['right', 'top']].set_visible(False)
axs[0,1].set_ylim(-1, 50)
axs[0,1].scatter(np.arange(p), hits[1], color='purple', marker="d", label="Simulated (observed)")
#axs[0,1].scatter(np.arange(p), h_ppc_low.mean(axis=1), color='g', alpha=0.7, label="Posterior Mean")
axs[0,1].errorbar(np.arange(p), h_ppc_low.mean(axis=1), yerr=h_ppc_low.std(axis=1), fmt='go', alpha=0.5, label="Predicted")
axs[0,1].errorbar(x=None, y=None, color="w", label="SDI: "+str(SDI_hlm)+" "+str(SDI_hl_hdi))
axs[0,1].grid(alpha=0.2)
axs[0,1].legend()
axs[0,1].set_xlabel("Simulated Participants")
#axs[0,1].set_ylabel("d' (sensitivity) ")
axs[0,1].set_title("Group 2 (low)")
axs[0,1].spines[['right', 'top']].set_visible(False)
axs[1,0].set_ylim(-1, 70)
axs[1,0].scatter(np.arange(p), fas[0], color='purple', marker="d", label="Simulated (observed)")
#axs[1,0].scatter(np.arange(p), f_ppc_high.mean(axis=1), color='g', alpha=0.7, label="Posterior Mean")
axs[1,0].errorbar(np.arange(p), f_ppc_high.mean(axis=1), yerr=f_ppc_high.std(axis=1), fmt='go', color='g', alpha=0.5, label="Predicted")
axs[1,0].errorbar(x=None, y=None, color="w", label="SDI: "+str(SDI_fhm)+" "+str(SDI_fh_hdi))
axs[1,0].grid(alpha=0.2)
axs[1,0].legend()
axs[1,0].set_xlabel("Simulated Participants")
axs[1,0].set_ylabel("False Alarms", size=18)
#axs[1,0].set_title("Group 1 (high)")
axs[1,0].spines[['right', 'top']].set_visible(False)
axs[1,1].set_ylim(-1, 70)
axs[1,1].scatter(np.arange(p), fas[1], color='purple', marker="d", label="Simulated (observed)")
#axs[1,1].scatter(np.arange(p), f_ppc_low.mean(axis=1), color='g', alpha=0.7, label="Posterior Mean")
axs[1,1].errorbar(np.arange(p), f_ppc_low.mean(axis=1), yerr=f_ppc_low.std(axis=1), fmt='go', color='g', alpha=0.5, label="Predicted")
axs[1,1].errorbar(x=None, y=None, color="w", label="SDI: "+str(SDI_flm)+" "+str(SDI_fl_hdi))
axs[1,1].grid(alpha=0.2)
axs[1,1].legend()
axs[1,1].set_xlabel("Simulated Participants")
#axs[1,1].set_ylabel("c (bias) ")
#axs[1,1].set_title("Group 2 (low)")
axs[1,1].spines[['right', 'top']].set_visible(False)
axs[0,0].text(s="B", x=-20, y=60, size=24)
axs[0,0].text(s="Model 2 (LKJ Model)", x=-10, y=60, size=20)
plt.tight_layout()
plt.savefig("mod2_posterior_predictives.png", dpi=800)
plt.show()



### participant average summary