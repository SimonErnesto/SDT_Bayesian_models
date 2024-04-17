# -*- coding: utf-8 -*-
import os
import pymc as pm
import pytensor.tensor as at
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

os.chdir(os.getcwd())

cdf = stats.norm.cdf
inv_cdf = stats.norm.ppf
pdf = stats.norm.pdf
        
np.random.seed(33)


#####plotting parameters
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.titlesize': 16})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"


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
    idata_base = pm.sample(1000, random_seed=33, nuts_sampler='numpyro')

ener_base = az.plot_energy(idata_base)
ener_base
plt.savefig("mo1_energy.png", dpi=300)    

base_summ = az.summary(idata_base, hdi_prob=0.9)
base_summ.to_csv("mod1_base.csv")


fig, ax = plt.subplots(4,2, figsize=(10,10))
az.plot_trace(idata_base.posterior, var_names=["d", "c", "H", "F"], kind="rank_vlines", axes=ax)
ax[0,0].title.set(fontsize=18)
ax[0,1].title.set(fontsize=18)
ax[1,0].title.set(fontsize=18)
ax[1,1].title.set(fontsize=18)
ax[2,0].title.set(fontsize=18)
ax[2,1].title.set(fontsize=18)
ax[3,0].title.set(fontsize=18)
ax[3,1].title.set(fontsize=18)
ax[0,0].xaxis.set_tick_params(labelsize=16)
ax[0,1].xaxis.set_tick_params(labelsize=16)
ax[1,0].xaxis.set_tick_params(labelsize=16)
ax[1,1].xaxis.set_tick_params(labelsize=16)
ax[2,0].xaxis.set_tick_params(labelsize=16)
ax[2,1].xaxis.set_tick_params(labelsize=16)
ax[3,0].xaxis.set_tick_params(labelsize=16)
ax[3,1].xaxis.set_tick_params(labelsize=16)
plt.suptitle("A  Model 1 (Base Model)", x=0.2)
plt.tight_layout()
plt.savefig("mod1_rank_plots.png", dpi=800)
plt.show()
plt.close()



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


ener_var = az.plot_energy(idata_var)
ener_var
plt.savefig("mod2_energy.png", dpi=300)    

var_summ = az.summary(idata_var, hdi_prob=0.9)
var_summ.to_csv("mod2_summary.csv")

fig, ax = plt.subplots(4,2, figsize=(10,10))
az.plot_trace(idata_var.posterior, var_names=["d", "c", "H", "F"], kind="rank_vlines", axes=ax)
ax[0,0].title.set(fontsize=18)
ax[0,1].title.set(fontsize=18)
ax[1,0].title.set(fontsize=18)
ax[1,1].title.set(fontsize=18)
ax[2,0].title.set(fontsize=18)
ax[2,1].title.set(fontsize=18)
ax[3,0].title.set(fontsize=18)
ax[3,1].title.set(fontsize=18)
ax[0,0].xaxis.set_tick_params(labelsize=16)
ax[0,1].xaxis.set_tick_params(labelsize=16)
ax[1,0].xaxis.set_tick_params(labelsize=16)
ax[1,1].xaxis.set_tick_params(labelsize=16)
ax[2,0].xaxis.set_tick_params(labelsize=16)
ax[2,1].xaxis.set_tick_params(labelsize=16)
ax[3,0].xaxis.set_tick_params(labelsize=16)
ax[3,1].xaxis.set_tick_params(labelsize=16)
plt.suptitle("B  Model 2 (Varying Model)", x=0.2)
plt.tight_layout()
plt.savefig("mod2_rank_plots.png", dpi=800)
plt.show()
plt.close()



# multilevel Model with LKJ correlations between d and c
with pm.Model() as mod_lkj:
    
    rho_high = pm.LKJCorr("ρ high", n=g, eta=2.0)
    d_high_sd = pm.HalfNormal("d_high_sd", 1.0)
    c_high_sd = pm.HalfNormal("c_high_sd", 1.0)
    d_high_mean = pm.Normal("d_high_mean", 0, 1.0)
    c_high_mean = pm.Normal("c_high_mean", 0, 1.0)
    high_cov = at.stack([[d_high_sd**2, rho_high[0] * d_high_sd * c_high_sd],
                         [rho_high[0] * d_high_sd * c_high_sd, c_high_sd**2]])
    high_means = at.stack([d_high_mean, c_high_mean])
    high = pm.MvNormal("high", mu=high_means, cov=high_cov, shape=(p,g))
    cov_high = pm.Deterministic("cov_high", high_cov)
    
    rho_low = pm.LKJCorr("ρ low", n=g, eta=2.0)
    d_low_sd = pm.HalfNormal("d_low_sd", 1.0)
    c_low_sd = pm.HalfNormal("c_low_sd", 1.0)
    d_low_mean = pm.Normal("d_low_mean", 0, 1.0)
    c_low_mean = pm.Normal("c_low_mean", 0, 1.0)
    low_cov = at.stack([[d_low_sd**2, rho_low[0] * d_low_sd * c_low_sd],
                         [rho_low[0] * d_low_sd * c_low_sd, c_low_sd**2]])
    low_means = at.stack([d_low_mean, c_low_mean])
    low = pm.MvNormal("low", mu=low_means, cov=low_cov, shape=(p,g))
    cov_low = pm.Deterministic("cov_low", low_cov)
    
    d = pm.Deterministic("d", at.stack([high[:,0], low[:,0]]))
    c = pm.Deterministic("c", at.stack([high[:,1], low[:,1]]))
    
    H = pm.Deterministic('H', Phi(0.5*d - c)) # hit rate 
    F = pm.Deterministic('F', Phi(-0.5*d - c)) # false alarm rate
    
    yh = pm.Binomial('yh', p=H, n=sig, observed=hits) # sampling for Hits, S is number of signal trials
    yf = pm.Binomial('yf', p=F, n=noi, observed=fas) # sampling for FAs, N is number of noise trials
  
with mod_lkj:
    idata_lkj = pm.sample(1000, random_seed=33, nuts_sampler='numpyro')

ener_lkj = az.plot_energy(idata_lkj)
ener_lkj
plt.savefig("mod3_energy.png", dpi=300)    

lkj_summ = az.summary(idata_lkj, hdi_prob=0.9)
lkj_summ.to_csv("mod3_summary.csv")

fig, ax = plt.subplots(6,2, figsize=(10,14))
az.plot_trace(idata_lkj.posterior, var_names=["ρ high", "ρ low", "d", "c", "H", "F"], 
              kind="rank_vlines", axes=ax)
ax[0,0].title.set(fontsize=18)
ax[0,1].title.set(fontsize=18)
ax[1,0].title.set(fontsize=18)
ax[1,1].title.set(fontsize=18)
ax[2,0].title.set(fontsize=18)
ax[2,1].title.set(fontsize=18)
ax[3,0].title.set(fontsize=18)
ax[3,1].title.set(fontsize=18)
ax[4,0].title.set(fontsize=18)
ax[4,1].title.set(fontsize=18)
ax[5,0].title.set(fontsize=18)
ax[5,1].title.set(fontsize=18)
ax[0,0].xaxis.set_tick_params(labelsize=16)
ax[0,1].xaxis.set_tick_params(labelsize=16)
ax[1,0].xaxis.set_tick_params(labelsize=16)
ax[1,1].xaxis.set_tick_params(labelsize=16)
ax[2,0].xaxis.set_tick_params(labelsize=16)
ax[2,1].xaxis.set_tick_params(labelsize=16)
ax[3,0].xaxis.set_tick_params(labelsize=16)
ax[3,1].xaxis.set_tick_params(labelsize=16)
ax[4,0].xaxis.set_tick_params(labelsize=16)
ax[4,1].xaxis.set_tick_params(labelsize=16)
ax[5,0].xaxis.set_tick_params(labelsize=16)
ax[5,1].xaxis.set_tick_params(labelsize=16)
plt.suptitle("C  Model 3 (LKJ Model)", x=0.2)
plt.tight_layout()
plt.savefig("mod3_rank_plots.png", dpi=800)
plt.show()
plt.close()
