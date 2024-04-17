# -*- coding: utf-8 -*-
import os
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import pickle
from scipy import stats
import matplotlib.pyplot as plt

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


signal = np.array([np.repeat(25, p), np.repeat(25, p)]) #fixed number of signal trials (25) 
noise = np.array([np.repeat(75, p), np.repeat(75, p)]) #fixed number of noise trials (75)

d_prime = np.array([d_high, d_low])
c_bias = np.array([c_high, c_low])

hit = np.random.binomial(signal, cdf(0.5*d_prime - c_bias)) #derive hits from d' and c
fa = np.random.binomial(noise, cdf(-0.5*d_prime - c_bias)) #derive false alarms from d' and c

print("Correlation coefficient low sensitivity:", correlation_high)
print("Correlation coefficient high sensitivity:", correlation_low)


# cumulative density of standard normal CDF a.k.a Phi
def Phi(x):
    #Cumulative distribution function of standard Gaussian
    return 0.5 + 0.5 * pm.math.erf(x / pm.math.sqrt(2))

def compile_mod(hit, signal, fa, noise):
    coords = {"obs_id": np.arange(len(hit.flatten()))}
    ##basic model
    with pm.Model(coords=coords) as model:
        
        hit_id = pm.Data("hit", hit.flatten(), dims="obs_id")
        fa_id = pm.Data("fa", fa.flatten(), dims="obs_id")
        
        d = pm.Normal('d', 0.0, 1.0, shape=hit.shape) #discriminability d'
        
        c = pm.Normal('c', 0.0, 1.0, shape=hit.shape) #bias c
        
        H = pm.Deterministic('H', Phi(0.5*d - c)).flatten() # hit rate
        F = pm.Deterministic('F', Phi(-0.5*d - c)).flatten() # false alarm rate
        
        yh = pm.Binomial('yh', p=H, n=signal.flatten(), observed=hit_id, dims='obs_id') # sampling for Hits, S is number of signal trials
        yf = pm.Binomial('yf', p=F, n=noise.flatten(), observed=fa_id, dims='obs_id') # sampling for FAs, N is number of noise trials
            
    return model


### define sampling arguments
sample_kwargs = {"draws":1000, "tune":1000, "chains":4, "cores":12, "random_seed":33, 'nuts_sampler':'numpyro'}#, "cores":1, "init":'advi'}
with compile_mod(hit,signal,fa,noise) as mod:
    idata = pm.sample(**sample_kwargs, idata_kwargs={"log_likelihood": True} )

dims = {"y": ["time"]}
idata_kwargs = {
    "dims": dims,
}



idata.log_likelihood['y'] = idata.log_likelihood['yh']+idata.log_likelihood['yf']
idata.log_likelihood = idata.log_likelihood.drop(['yh', 'yf'])


loo_orig = az.loo(idata, pointwise=True)
loo_orig


####### This wrapper function is taken literally from Arviz just to make easier
####### checking up index and arrays dimensions 
"""Stats functions that require refitting the model."""
import logging

import numpy as np

from arviz import loo
from arviz.stats.stats_utils import logsumexp as _logsumexp

__all__ = ["reloo"]

_log = logging.getLogger(__name__)


def reloo(wrapper, loo_orig=None, k_thresh=0.7, scale=None, verbose=True):

    required_methods = ("sel_observations", "sample", "get_inference_data", "log_likelihood__i")
    not_implemented = wrapper.check_implemented_methods(required_methods)
    if not_implemented:
        raise TypeError(
            "Passed wrapper instance does not implement all methods required for reloo "
            f"to work. Check the documentation of SamplingWrapper. {not_implemented} must be "
            "implemented and were not found."
        )
    if loo_orig is None:
        loo_orig = loo(wrapper.idata_orig, pointwise=True, scale=scale)
    loo_refitted = loo_orig.copy()
    khats = loo_refitted.pareto_k
    print('khats: '+str(khats.shape))
    loo_i = loo_refitted.loo_i
    scale = loo_orig.scale

    if scale.lower() == "deviance":
        scale_value = -2
    elif scale.lower() == "log":
        scale_value = 1
    elif scale.lower() == "negative_log":
        scale_value = -1
    lppd_orig = loo_orig.p_loo + loo_orig.elpd_loo / scale_value
    n_data_points = loo_orig.n_data_points

    # if verbose:
    #     warnings.warn("reloo is an experimental and untested feature", UserWarning)

    if np.any(khats > k_thresh):
        #print('index: '+str(idx))
        
        for idx in np.argwhere(khats.values > 0.7):
            if verbose:
                _log.info("Refitting model excluding observation %d", idx)
            new_obs, excluded_obs = wrapper.sel_observations(idx)
            fit = wrapper.sample(new_obs)
            idata_idx = wrapper.get_inference_data(fit)
            log_like_idx = wrapper.log_likelihood__i(excluded_obs, idata_idx).values.flatten()
            loo_lppd_idx = scale_value * _logsumexp(log_like_idx, b_inv=len(log_like_idx))
            khats[idx] = 0
            loo_i[idx] = loo_lppd_idx
        loo_refitted.loo = loo_i.values.sum()
        loo_refitted.loo_se = (n_data_points * np.var(loo_i.values)) ** 0.5
        loo_refitted.p_loo = lppd_orig - loo_refitted.loo / scale_value
        return loo_refitted
    else:
        _log.info("No problematic observations")
        return loo_orig


class Wrapper(az.SamplingWrapper):
    def __init__(self, hit, signal, fa, noise, **kwargs):
        super(Wrapper, self).__init__(**kwargs)

        self.hit = hit
        self.signal = signal
        self.fa = fa
        self.noise = noise
        
    def sample(self, modified_observed_data):
        with self.model(**modified_observed_data) as mod:
            idata = pm.sample(
                **self.sample_kwargs, 
                return_inferencedata=True, 
                idata_kwargs={"log_likelihood": True} )
            
        self.pymc3_model = mod
        idata.log_likelihood['y'] = idata.log_likelihood['yh']+idata.log_likelihood['yf']
        idata.log_likelihood = idata.log_likelihood.drop(['yh', 'yf'])
        #idata.log_likelihood['y'] = idata.log_likelihood['y'].mean(axis=3)
        idata = idata.log_likelihood['y']
        return idata
    
    def get_inference_data(self, idata):
        #idata = az.from_pymc3(trace, model=self.pymc3_model, **self.idata_kwargs)
        #idata.pymc3_trace = trace
        return idata
        
    def log_likelihood__i(self, excluded_observed_data, idata__i):
        log_lik__i = idata.log_likelihood['y']
        #print(log_lik__i)
        return log_lik__i
        
    def sel_observations(self, idx):
        print("index: "+str(idx))
        
        mask = np.isin(np.arange(len(self.signal.flatten())), idx)
        
        sigi = np.array([ self.signal.flatten()[~mask] ])
        hi = np.array([ self.hit.flatten()[~mask] ])
        noisi = np.array([ self.noise.flatten()[~mask] ])
        fai = np.array([ self.fa.flatten()[~mask] ])
        
        sige = np.array([ self.signal.flatten()[mask] ])
        he = np.array([ self.hit.flatten()[mask] ])
        noise = np.array([ self.noise.flatten()[~mask] ])
        fae = np.array([ self.fa.flatten()[~mask] ])
        
        data__i = {"signal":sigi, "hit":hi, 'fa':fai, 'noise':noisi}
        data_ex = {"signal":sige, "hit":he, 'fa':fae, 'noise':noise}

        return data__i, data_ex

wrapper = Wrapper(model=compile_mod, 
                  hit=hit, 
                  signal=signal,  
                  fa=fa,
                  noise=noise,
                  sample_kwargs=sample_kwargs, 
                  idata_kwargs=idata_kwargs)

loo_relooed = reloo(wrapper, loo_orig=loo_orig)
loo_relooed

reloo_df = pd.DataFrame(loo_relooed)
reloo_df.to_csv("mod1_reloo.csv")

reloo_k = loo_relooed.pareto_k
k_df = pd.DataFrame(reloo_k)
k_df.to_csv("mod1_k_hats.csv")
az.plot_khat(loo_relooed)
plt.savefig('mod1_khat.png', dpi=300)
plt.close()

loo_i = loo_relooed.loo_i
np.save("mod1_loo_i.npy", loo_i)

file = open("mod1_reloo.obj","wb")
pickle.dump(loo_relooed,file)
file.close()

log = open("mod1_reloo_summary.txt", 'a')
def oprint(message):
    print(message)
    global log
    log.write(message)
    return()
oprint(str(loo_relooed))
log.close()

log = open("mod1_loo_orig_summary.txt", 'a')
def oprint(message):
    print(message)
    global log
    log.write(message)
    return()
oprint(str(loo_orig))
log.close()