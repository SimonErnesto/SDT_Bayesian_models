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


# basic Model
with pm.Model() as mod_base:
    
    d = pm.Normal('d', 0.0, 1, shape=(g,p)) #discriminability d'
    
    c = pm.Normal('c', 0.0, 1, shape=(g,p)) #bias c
    
    H = pm.Deterministic('H', Phi(0.5*d - c)) # hit rate
    F = pm.Deterministic('F', Phi(-0.5*d - c)) # false alarm rate
    
    yh = pm.Binomial('yh', p=H, n=sig, observed=hits) # sampling for Hits, sig is number of signal trials
    yf = pm.Binomial('yf', p=F, n=noi, observed=fas) # sampling for FAs, noi is number of noise trials

with mod_base:
    ppc_base = pm.sample_prior_predictive(1000, random_seed=33)

base_summ = az.summary(ppc_base, hdi_prob=0.9)    
base_summ_like = az.summary(ppc_base.prior_predictive, hdi_prob=0.9)   
base_summ = pd.concat([base_summ, base_summ_like]) 
base_summ.to_csv("mod1_prior_predictive_summary.csv")

ppc_base = ppc_base.stack(sample = ['chain', 'draw']).prior_predictive


    
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
    ppc_var = pm.sample_prior_predictive(1000, random_seed=33)
 
var_summ = az.summary(ppc_var, hdi_prob=0.9)    
var_summ_like = az.summary(ppc_var.prior_predictive, hdi_prob=0.9)   
var_summ = pd.concat([var_summ, var_summ_like]) 
var_summ.to_csv("mod2_prior_predictive_summary.csv")

ppc_var = ppc_var.stack(sample = ['chain', 'draw']).prior_predictive


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
    ppc_lkj = pm.sample_prior_predictive(1000, random_seed=33)
 
lkj_summ = az.summary(ppc_lkj, hdi_prob=0.9)    
lkj_summ_like = az.summary(ppc_lkj.prior_predictive, hdi_prob=0.9)   
lkj_summ = pd.concat([lkj_summ, lkj_summ_like]) 
lkj_summ.to_csv("mod3_prior_predictive_summary.csv")
     
ppc_lkj = ppc_lkj.stack(sample = ['chain', 'draw']).prior_predictive



###############################################################################

##plotting paramters
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'figure.titlesize': 18})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.markerfacecolor'] = 'w'
plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.markeredgewidth'] = 2


###### Plot prior predictive for base model (model 1)
h_base = ppc_base['yh'].values #predicted hits
f_base = ppc_base['yf'].values #predicted false alarms

fig, axs = plt.subplots(2, 2, figsize=(10,10))

axs[0,0].set_ylim(0,30)
axs[0,0].hist(hits[0], color="purple", bins=100, label="Observed")
samples = np.random.choice(np.arange(h_base[0].shape[1]), 50)
for s in samples:
    axs[0,0].hist(h_base[0,:,s], color="g", alpha=0.01, bins=100) 
axs[0,0].hist(h_base[0,:,1], color="g", alpha=0.2, bins=100, label="Predicted") 
axs[0,0].set_xlabel("Hits")
axs[0,0].set_ylabel("Frequency")
axs[0,0].legend(loc="upper center")
axs[0,0].set_title("Group 1 (high)")

axs[0,1].set_ylim(0,30)
axs[0,1].hist(hits[1], color="purple", bins=100, label="Observed")
samples = np.random.choice(np.arange(h_base[1].shape[1]), 50)
for s in samples:
    axs[0,1].hist(h_base[1,:,s], color="g", alpha=0.01, bins=100) 
axs[0,1].hist(h_base[1,:,1], color="g", alpha=0.2, bins=100, label="Predicted") 
axs[0,1].set_xlabel("Hits")
axs[0,1].legend(loc="upper center")
axs[0,1].set_title("Group 2 (low)")

axs[1,0].set_ylim(0,30)
axs[1,0].hist(fas[0], color="purple", bins=100, label="Observed")
samples = np.random.choice(np.arange(f_base[0].shape[1]), 50)
for s in samples:
    axs[1,0].hist(f_base[0,:,s], color="g", alpha=0.01, bins=100) 
axs[1,0].hist(f_base[0,:,1], color="g", alpha=0.2, bins=100, label="Predicted") 
axs[1,0].set_xlabel("False Alarms")
axs[1,0].set_ylabel("Frequency")
axs[1,0].legend(loc="upper center")
#axs[1,0].set_title("Group 1 (high)")

axs[1,1].set_ylim(0,30)
axs[1,1].hist(fas[1], color="purple", bins=100, label="Observed")
samples = np.random.choice(np.arange(f_base[1].shape[1]), 50)
for s in samples:
    axs[1,1].hist(f_base[1,:,s], color="g", alpha=0.01, bins=100) 
axs[1,1].hist(f_base[1,:,1], color="g", alpha=0.2, bins=100, label="Predicted") 
axs[1,1].set_xlabel("False Alarms")
axs[1,1].legend(loc="upper center")
#axs[1,1].set_title("Group 2 (low)")

axs[0,0].text(s="A", x=-5, y=35, size=24)
axs[0,0].text(s="Model 1 (Base Model)", x=-1, y=35, size=22)

plt.tight_layout()
plt.savefig("mod1_prior_predictives.png", dpi=800)
plt.show()
plt.close()



###### Plot prior predictive for varying model (model 2)
h_var = ppc_var['yh'].values #predicted hits
f_var = ppc_var['yf'].values #predicted false alarms


fig, axs = plt.subplots(2, 2, figsize=(10,10))

axs[0,0].set_ylim(0,30)
axs[0,0].hist(hits[0], color="purple", bins=100, label="Observed")
samples = np.random.choice(np.arange(h_var[0].shape[1]), 50)
for s in samples:
    axs[0,0].hist(h_var[0,:,s], color="g", alpha=0.01, bins=100) 
axs[0,0].hist(h_var[0,:,1], color="g", alpha=0.2, bins=100, label="Predicted") 
axs[0,0].set_xlabel("Hits")
axs[0,0].set_ylabel("Frequency")
axs[0,0].legend(loc="upper center")
axs[0,0].set_title("Group 1 (high)")

axs[0,1].set_ylim(0,30)
axs[0,1].hist(hits[1], color="purple", bins=100, label="Observed")
samples = np.random.choice(np.arange(h_var[1].shape[1]), 50)
for s in samples:
    axs[0,1].hist(h_var[1,:,s], color="g", alpha=0.01, bins=100) 
axs[0,1].hist(h_var[1,:,1], color="g", alpha=0.2, bins=100, label="Predicted") 
axs[0,1].set_xlabel("Hits")
axs[0,1].legend(loc="upper center")
axs[0,1].set_title("Group 2 (low)")

axs[1,0].set_ylim(0,30)
axs[1,0].hist(fas[0], color="purple", bins=100, label="Observed")
samples = np.random.choice(np.arange(f_var[0].shape[1]), 50)
for s in samples:
    axs[1,0].hist(f_var[0,:,s], color="g", alpha=0.01, bins=100) 
axs[1,0].hist(f_var[0,:,1], color="g", alpha=0.2, bins=100, label="Predicted") 
axs[1,0].set_xlabel("False Alarms")
axs[1,0].set_ylabel("Frequency")
axs[1,0].legend(loc="upper center")
#axs[1,0].set_title("Group 1 (high)")

axs[1,1].set_ylim(0,30)
axs[1,1].hist(fas[1], color="purple", bins=100, label="Observed")
samples = np.random.choice(np.arange(f_var[1].shape[1]), 50)
for s in samples:
    axs[1,1].hist(f_var[1,:,s], color="g", alpha=0.01, bins=100) 
axs[1,1].hist(f_var[1,:,1], color="g", alpha=0.2, bins=100, label="Predicted") 
axs[1,1].set_xlabel("False Alarms")
axs[1,1].legend(loc="upper center")
#axs[1,1].set_title("Group 2 (low)")

axs[0,0].text(s="B", x=-5, y=35, size=24)
axs[0,0].text(s="Model 2 (Varying Model)", x=-1, y=35, size=22)

plt.tight_layout()
plt.savefig("mod2_prior_predictives.png", dpi=800)
plt.show()
plt.close()




###### Plot prior predictive for LKJ model (model 3)
h_lkj = ppc_lkj['yh'].values #predicted hits
f_lkj = ppc_lkj['yf'].values #predicted false alarms

fig, axs = plt.subplots(2, 2, figsize=(10,10))

axs[0,0].set_ylim(0,30)
axs[0,0].hist(hits[0], color="purple", bins=100, label="Observed")
samples = np.random.choice(np.arange(h_lkj[0].shape[1]), 50)
for s in samples:
    axs[0,0].hist(h_lkj[0,:,s], color="g", alpha=0.01, bins=100) 
axs[0,0].hist(h_lkj[0,:,1], color="g", alpha=0.2, bins=100, label="Predicted") 
axs[0,0].set_xlabel("Hits")
axs[0,0].set_ylabel("Frequency")
axs[0,0].legend(loc="upper center")
axs[0,0].set_title("Group 1 (high)")

axs[0,1].set_ylim(0,30)
axs[0,1].hist(hits[1], color="purple", bins=100, label="Observed")
samples = np.random.choice(np.arange(h_lkj[1].shape[1]), 50)
for s in samples:
    axs[0,1].hist(h_lkj[1,:,s], color="g", alpha=0.01, bins=100) 
axs[0,1].hist(h_lkj[1,:,1], color="g", alpha=0.2, bins=100, label="Predicted") 
axs[0,1].set_xlabel("Hits")
axs[0,1].legend(loc="upper center")
axs[0,1].set_title("Group 2 (low)")

axs[1,0].set_ylim(0,30)
axs[1,0].hist(fas[0], color="purple", bins=100, label="Observed")
samples = np.random.choice(np.arange(f_lkj[0].shape[1]), 50)
for s in samples:
    axs[1,0].hist(f_lkj[0,:,s], color="g", alpha=0.01, bins=100) 
axs[1,0].hist(f_lkj[0,:,1], color="g", alpha=0.2, bins=100, label="Predicted") 
axs[1,0].set_xlabel("False Alarms")
axs[1,0].set_ylabel("Frequency")
axs[1,0].legend(loc="upper center")
#axs[1,0].set_title("Group 1 (high)")

axs[1,1].set_ylim(0,30)
axs[1,1].hist(fas[1], color="purple", bins=100, label="Observed")
samples = np.random.choice(np.arange(f_lkj[1].shape[1]), 50)
for s in samples:
    axs[1,1].hist(f_lkj[1,:,s], color="g", alpha=0.01, bins=100) 
axs[1,1].hist(f_lkj[1,:,1], color="g", alpha=0.2, bins=100, label="Predicted") 
axs[1,1].set_xlabel("False Alarms")
axs[1,1].legend(loc="upper center")
#axs[1,1].set_title("Group 2 (low)")

axs[0,0].text(s="C", x=-5, y=35, size=24)
axs[0,0].text(s="Model 3 (LKJ Model)", x=-1, y=35, size=22)

plt.tight_layout()
plt.savefig("mod3_prior_predictives.png", dpi=800)
plt.show()
plt.close()

