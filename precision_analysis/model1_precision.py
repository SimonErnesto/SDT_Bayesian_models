# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
import pymc as pm
import arviz as az
import pandas as pd

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

ps = np.arange(0, 100, 10)

d_high_m = []
d_high_h5 = []
d_high_h95 = []
precis_d_high = []

d_low_m = []
d_low_h5 = []
d_low_h95 = []
precis_d_low = []

c_high_m = []
c_high_h5 = []
c_high_h95 = []
precis_c_high = []

c_low_m = []
c_low_h5 = []
c_low_h95 = []
precis_c_low = []

obs_d_high = []
obs_d_low = []
obs_c_high = []
obs_c_low = []


for i in range(len(ps)):
    
    pi = ps[i]
    
    j = i+1
        
    # basic model
    with pm.Model() as mod:
        
        d = pm.Normal('d', 0.0, 0.5, shape=(g,pi+10)) #discriminability d'
        
        c = pm.Normal('c', 0.0, 2.0, shape=(g,pi+10)) #bias c
        
        H = pm.Deterministic('H', Phi(0.5*d - c)) # hit rate
        F = pm.Deterministic('F', Phi(-0.5*d - c)) # false alarm rate
        
        yh = pm.Binomial('yh', p=H, n=sig[:,:pi+10], observed=hits[:,:pi+10]) # sampling for Hits, sig is number of signal trials
        yf = pm.Binomial('yf', p=F, n=noi[:,:pi+10], observed=fas[:,:pi+10]) # sampling for FAs, noi is number of noise trials

    with mod:
        idata = pm.sample(1000, chains=4, cores=12, random_seed=33)
        
    pos = idata.stack(sample = ['chain', 'draw']).posterior
    
    d_pos_high = pos['d'][0,:,:].values
    h5,h95 = az.hdi(d_pos_high.mean(axis=0), hdi_prob=0.9)
    d_high_m.append(d_pos_high.mean())
    d_high_h5.append(h5)
    d_high_h95.append(h95)

    d_pos_low = pos['d'][1,:,:].values
    h5,h95 = az.hdi(d_pos_low.mean(axis=0), hdi_prob=0.9)
    d_low_m.append(d_pos_low.mean())
    d_low_h5.append(h5)
    d_low_h95.append(h95)
    
    c_pos_high = pos['c'][0,:,:].values
    h5,h95 = az.hdi(c_pos_high.mean(axis=0), hdi_prob=0.9)
    c_high_m.append(c_pos_high.mean())
    c_high_h5.append(h5)
    c_high_h95.append(h95)

    c_pos_low = pos['c'][1,:,:].values
    h5,h95 = az.hdi(c_pos_low.mean(axis=0), hdi_prob=0.9)
    c_low_m.append(c_pos_low.mean())
    c_low_h5.append(h5)
    c_low_h95.append(h95)
    
    obs_d_high.append(d_high[:pi+10].mean())
    obs_d_low.append(d_low[:pi+10].mean())
    obs_c_high.append(c_high[:pi+10].mean())
    obs_c_low.append(c_low[:pi+10].mean())

    precis_d_high.append(abs(d_high_h95[i] - d_high_h5[i]))
    precis_d_low.append(abs(d_low_h95[i] - d_low_h5[i]))
    
    precis_c_high.append(abs(c_high_h95[i] - c_high_h5[i]))
    precis_c_low.append(abs(c_low_h95[i] - c_low_h5[i]))
    
    
    if j > 1:        
        pre_precs = np.array([abs(precis_d_high[i-1] - precis_d_high[i]), 
                              abs(precis_d_low[i-1] - precis_d_low[i]),
                              abs(precis_c_high[i-1] - precis_c_high[i]),
                              abs(precis_c_low[i-1] - precis_c_low[i])])
    if j < 2:
        pre_precs = np.array([1,1,1,1])
                              
    
    precs = np.array([precis_d_high[i],precis_d_low[i],
                      precis_c_high[i], precis_c_high[i]])
    

    print("Run #"+str(i))
    print("high d' precision: "+str(precis_d_high[i].round(2)))
    print("low d' precision: "+str(precis_d_low[i].round(2)))
    print("high c precision: "+str(precis_c_high[i].round(2)))
    print("low c precision: "+str(precis_c_low[i].round(2)))
    
    print("high d' precision diff: "+str(pre_precs[0].round(2)))
    print("low d' precision diff: "+str(pre_precs[1].round(2)))
    print("high c precision diff: "+str(pre_precs[2].round(2)))
    print("low c precision diff: "+str(pre_precs[3].round(2)))
        

df = pd.DataFrame({"d_high_m":d_high_m, "d_high_h5":d_high_h5, "d_high_h95":d_high_h95, 
                   "d_low_m":d_low_m, "d_low_h5":d_low_h5, "d_low_h95":d_low_h95, 
                   "c_high_m":c_high_m, "c_high_h5":c_high_h5, "c_high_h95":c_high_h95,
                   "c_low_m":c_low_m, "c_low_h5":c_low_h5, "c_low_h95":c_low_h95,
                   "obs_d_high":obs_d_high, "obs_d_low":obs_d_low,
                   "obs_c_high":obs_c_high, "obs_c_low":obs_c_low})
df.to_csv("mod1_precision_analysis.csv", index=False)