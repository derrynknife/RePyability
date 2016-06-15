import pandas as pd
import re
import matplotlib.pyplot as plt
import brewer2mpl
import scipy
import numpy as np
import nltk
from collections import Counter
from scipy.stats import t as qt
from scipy.optimize import minimize

### Need to ensure that l = -1
# f = 0, and 
# r = 1


### Non-Repairable Components

### Non-Parametric Methods

# For the non-parametric methods need to establish:

# x is the random domain
# r is the number of items at RISK at x
# c is the number of censored (right) at the time
# d is the number of failures at x
# N is the total number of components
# R is reliability/survival
# F is prob of failure
# h is hazard rate
# H is cum hazard rate
def get_r_c_d(x, censored, counts):
    # Rank computations for:
    # Items at risk
    # r = n(i-1) - d(i-1) - c(i-1)
    # Hazard Rate
    # hi = d/r
    # Cumulative Hazard Function
    # H = cumsum(h)
    # Reliability Function
    # R = exp(-H)
    if counts is None:
        N = len(x)
    else:
        N = np.sum(counts)

    if censored is None:
        # Uncensored data
        if counts is None:
            # If there are no counts, there may be repeated x-values.
            x, d = np.unique(x, return_counts=True)
        else:
            # Assumption: given that data is given as counts it is assumed
            # that there is no duplicate x's...
            idx = np.argsort(x)
            x = x[idx]
            d = counts[idx]
        c = np.zeros_like(x)
    else:
        # Censored Data
        if counts is None:
            # No counts, i.e., individual failure times
            t = x
            x = np.unique(t)
            d = np.zeros_like(x)
            c = np.zeros_like(x)

            for i in range(len(t)):
                d[x == t[i]] += 1-censored[i]
                c[x == t[i]] += censored[i]
        else:
            # Last Case - Counts and Censored
            idx = np.argsort(x)
            x = x[idx]
            counts = counts[idx]
            censored = censored[idx]
            t = x
            x = np.unique(t)
            d = np.zeros_like(x)
            c = np.zeros_like(x)

            for i in range(len(t)):
                d[x == t[i]] += (1-censored[i])*counts[i]
                c[x == t[i]] += censored[i]*counts[i]

    # Calculate the risk set
    r = np.repeat(N, len(d))
    r[1::] = r[1::] - np.cumsum(d)[0:-1] - np.cumsum(c)[0:-1]
    return x, r, c, d
def nelson_aalen(x, censored=None, counts=None):
    # Nelson-Aalen estimation of Reliability function
    # Nelson, W.: Theory and Applications of Hazard Plotting for Censored Failure Data. 
    # Technometrics, Vol. 14, #4, 1972
    # Technically the NA estimate is for the Cumulative Hazard Function,
    # The reliability (survival) curve that is output is also known as the Breslow estimate.
    # I will leave it as Nelson-Aalen for this library.
    x = np.array(x)
    if censored is not None: censored = np.array(censored)
    if counts is not None: counts = np.array(counts)

    # Assumes whole population given
    if counts is None:
        N = len(x)
    else:
        N = np.sum(counts)

    # Rank computations for:
    # Items at risk
    # r = n(i-1) - d(i-1) - c(i-1)
    # Hazard Rate
    # hi = d/r
    # Cumulative Hazard Function
    # H = cumsum(h)
    # Reliability Function
    # R = exp(-H)

    # Find unique x, risk set, right censoring count, and failure counts
    x, r, c, d = get_r_c_d(x, censored, counts)
    # Calculate the hazard rate
    h = (d/r)
    # Calculate the Cumulativ Hazard
    H = np.cumsum(h)
    # Reliability!
    R = np.exp(-H)

    # Return a Non-Parametric Object
    out = NonParametric()
    out.R = R
    out.F = 1 - R
    out.h = h
    out.H = H
    out.c = c
    out.x = x
    out.d = d
    out.N = N
    out.r = r
    out.model = "Nelson-Aalen"
    return out    
def kaplan_meier(x, censored=None, counts=None):
    # Kaplan-Meier estimate of survival
    # Good explanation of K-M reason can be found at:
    # http://reliawiki.org/index.php/Non-Parametric_Life_Data_Analysis#Kaplan-Meier_Example
    # Data given not necessarily in order
    # Assumes whole population given
    x = np.array(x)
    if censored is not None: censored = np.array(censored)
    if counts is not None: counts = np.array(counts)

    # Assumes whole population given
    if counts is None:
        N = len(x)
    else:
        N = np.sum(counts)

    # Find unique x, risk set, right censoring count, and failure counts
    x, r, c, d = get_r_c_d(x, censored, counts)
    # Get R, H, h and F for K-M estimate
    R = np.cumprod(np.divide((r - d), r))
    H = -np.log(R)
    h = np.zeros_like(H)
    h[0] = H[0]
    h[1::] = np.diff(H)

    out = NonParametric()
    out.R = R
    out.F = 1 - R
    out.H = H
    out.h = h
    out.c = c
    out.x = x
    out.r = r
    out.d = d
    out.N = N
    
    out.model = "Kaplan-Meier"
    
    return out
def fleming_harrington(x, censored=None, counts=None):
    # Fleming-Harrington estimation of Reliability function (via cum hazard)
    # Fleming, T. R., and Harrington, D. P. (1984). “Nonparametric Estimation of the Survival Distribution in Censored Data.” 
    # Communications in Statistics—Theory and Methods 13:2469–2486.
    # Equation taken from http://support.sas.com/documentation/cdl/en/statug/68162...
    # .../HTML/default/viewer.htm#statug_lifetest_details03.htm

    x = np.array(x)
    if censored is not None: censored = np.array(censored)
    if counts is not None: counts = np.array(counts)

    # Assumes whole population given
    if counts is None:
        N = len(x)
    else:
        N = np.sum(counts)
    
    # Find unique x, risk set, right censoring count, and failure counts
    x, r, c, d = get_r_c_d(x, censored, counts)

    h = [np.sum([1./(r[i]-j) for j in range(d[i])]) for i in range(len(x))]
    H = np.cumsum(h)
    R = np.exp(-H)
    F = 1 - R

    out = NonParametric()
    out.R = R
    out.F = F
    out.d = d
    out.h = h
    out.r = r
    out.H = H
    out.c = c
    out.x = x
    out.N = N
    out.model = "Fleming-Harrington"
    
    return out
def turnbull(lower, upper, count=None):
    lower = np.array(lower)
    upper = np.array(upper)

    # Component Number, n
    n = len(lower)

    x = np.append(lower, upper)
    x = np.unique(x)
    m = len(x)
    p = np.repeat(1./m, m)
    S = 1. - np.cumsum(p)

    d = np.zeros(m)

    alphas = np.zeros((n, m))
    for j in range(m):
        for i in range(n):
            alphas[i, j] = int((lower[i] <= x[j]) and (upper[i] >= x[j]))

    for j in range(m):
        dd = np.zeros(n)
        for i in range(n):
            numerator = alphas[i, j] * p[j]
            denominator = 0
            for k in range(m):
                denominator += alphas[i, k] * p[k]
            dd[i] = numerator / denominator
        d[j] = np.sum(dd)

    return alphas, d
def success_run(n, confidence=0.95, alpha=None):
    if alpha is None: alpha = 1 - confidence
    return np.power(alpha, 1./n)
class NonParametric():
    '''
    This is a class used to create an object that can be used to perform a variety 
    of Reliability Engineering tasks.

    The intent is to encapsulate some reliability functions to reduce the complexity.
    Need to have data for:
    CDF - F
    Rel - R
    PDF - f
    Haz - h
    Cumh- H
    dR  - Reliability Confidence bound.
    given x and c
    
    Model - KM, NA, or FH
    
    
    Need to have functions for:
    Estimate F, R, f, h, H for any x
    
    Confidence Bound Funciton
    
    '''
    def __init__(self):
        # Need:
        # F
        # R
        # h
        # H
        # f
        # dR
        self.R = None
        self.F = None
        self.h = None
        self.H = None
        self.f = None
        
        self.d = None
        self.N = None
        self.c = None
        self.x = None
        self._ddR = None
        
        self.model = None
        self._dR = None  
    def __str__(self):
        # Used to automate print(NonParametric()) call
        return "%s Reliability Model" % self.model
    def confidence_bounds(self, alpha=0.05):
        if self._dR is None:
            self._dR = qt.ppf(alpha/2, self.r[0] - 1)*self._ddR
        return (self.R + self._dR), (self.R - self._dR)
    def upper_confidence_bound(self, confidence=0.95, alpha=None):
        if alpha is None:
            alpha = 1 - confidence
        return self.R - qt.ppf(alpha/2, self.n[0] - 1)* self._ddR  
    def lower_confidence_bound(self, confidence=0.95, alpha=None):
        if alpha is None:
            alpha = 1 - confidence
        return self.R + qt.ppf(alpha/2, self.n[0] - 1)* self._ddR
    def plot(self, confidence_bounds=False):
        R = np.append(1, self.R)
        x = np.append(0, self.x)
        plt.step(x, R, where='post')
        plt.ylim(0, 1)
        plt.xlim(0, max(x))
        plt.show()
    def to_dataframe(self):
        data = {
            'x' : self.x,
            'Failures' : self.d,
            'Failure Prob' : self.F,
            'Reliability' : self.R,
            'Censored' : self.c,
            'Risk Set' : self.r,
            'Hazard Rate' : self.h,
            'Cumulative Haz' : self.H
        }    
        return pd.DataFrame(data)
#--- Parametric Models

### Need to have prob models.
def plotting_positions(t, censored=None, formula="Blom"):
    # Numbers from "Effect of Renking Selection on the Weibull Modulus Estimation"
    # Authors: Kirtay, S; Dispinar, D.
    # From: Gazi University Journal of Science 25(1):175-187, 2012.
    # Assumes no repeated times.

    # Adust ranks if censored data present
    if censored is not None:
        ranks = rank_adjust(t, censored)
    else:
        ranks = np.array(range(1, len(t) + 1))

    # Set values for plotting position adjustments
    # Some repeated models with different names (for readability)
    if   formula == "Blom":       A, B = 0.375, 0.25
    elif formula == "Median":     A, B = 0.3, 0.4
    elif formula == "Modal":      A, B = 1.0, -1.0
    elif formula == "Midpoint":   A, B = 0.5, 0.0
    elif formula == "Mean":       A, B = 0.0, 1.0
    elif formula == "Weibull":    A, B = 0.0, 1.0
    elif formula == "Benard":     A, B = 0.3, 0.2
    elif formula == "Beard":      A, B = 0.31, 0.38
    elif formula == "Hazen":      A, B = 0.5, 0.0
    elif formula == "Filiben":    A, B = 0.3175, 1.635
    elif formula == "Gringorten": A, B = 0.44, 0.12
    elif formula == "None":       A, B = 0.0, 0.0
    elif formula == "Tukey":      A, B = 1./3., 1./3.
    elif formula == "DPW":        A, B = 1.0, 0.0

    # Use general adjustment formula
    pp = (ranks - A)/(len(ranks) + B)
    return pp
def rank_adjust(t, censored=None):
    # Currently limited to only Mean Order Number
    # Room to expand to:
    # Mode Order Number, and
    # Median Order Number
    # Uses mean order statistic to conduct rank adjustment
    # For further reading see:
    # http://reliawiki.org/index.php/Parameter_Estimation
    # Above reference provides excellent explanation of how this method is derived
    # This function currently assumes good input - Use if you know how
    # 15 Mar 2015

    # Total items in test/population
    n = len(t)
    # Preallocate adjusted ranks array
    ranks = np.zeros(n)
    
    if censored is None:
        censored = np.zeros(n)

    # Rank increment for [right] censored data
    # Previous Mean Order Number
    PMON = 0
    
    # Implemented in loop:
    # "Number of Items Before Present Suspended Set"
    # NIBPSS = n - (i - 1)
    # Denominator of rank increment = 1 + NIBPSS = n - i + 2
    for i in range(0, n):
        if censored[i] == 0:
            ranks[i] = PMON + (n + 1 - PMON)/(n - i + 2)
            PMON = ranks[i]
        else:
            ranks[i] = np.nan
    # Return adjusted ranks
    return ranks
def weibull_lsq(t, censored=None, plotting="Blom"):
    # This currently performs only RRY
    # Need to add RRX - simple by only adding np.polyfit(y, x, 1)
    # Fits a Weibull model using cumulative probability and times (or stress) to failure.
    if len(t) == 0:
        return None
    if censored is None:
        censored = np.zeros(len(t))
    t = np.array(t)
    t = sorted(t)
    f = plotting_positions(t, censored, plotting)
    # Convert data to linearised form
    x = np.log(t[censored == 0])
    y = np.log(np.log(1/(1 - f[censored == 0])))
    
    # Fit a linear model to the data.
    model = np.polyfit(x, y, 1)
    
    # Compute alpha and beta from linearised least square model
    beta  = model[0]
    alpha = np.exp(model[1]/-beta)
    
    # Output calculated parameters
    return alpha, beta
def weibull_mle(t, censored=None):
    # Fits a Weibull model using cumulative probability and times (or stress) to failure.
    t = np.array(t)

    if len(t) == 0:
        return None
    if censored is None:
        censored = np.zeros(len(t))
    
    idx = np.argsort(t)
    t = t[idx]
    censored = censored[idx]
    # Create anonymous function to use with optimise
    fun = lambda x: -sum((1-censored)*(np.log(x[1]/x[0]) + 
        (x[1]-1)*np.log(t/x[0]) - (t/x[0])**(x[1])) - (censored)*((t/x[0])**(x[1])))
    
    # Set bounds for alpha, beta
    bounds = ((0, None), (0, None))
    
    # Fit a linear model to the data.
    res = minimize(fun, (np.mean(t), 1.0), bounds=bounds)

    alpha, beta = res.x

    return alpha, beta
def weibull_lfp_lsq(f, t, censored=None):
    # Fits a Weibull model using cumulative probability and times (or stress) to failure.
    if len(f) == 0 or len(t) == 0:
        return None
    if censored is None:
        censored = np.zeros(len(t))
        
    t = t[~np.isnan(f)]
    f = f[~np.isnan(f)]
    
    # Create anonymous function to use with optimise
    fun = lambda x: sum(((x[0] * (1 - np.exp(-(t/x[1])**x[2]))) - f)**2)
    
    # Set bounds for p, alpha, beta
    bounds = ((0, 1), (0, None), (0, None))
    
    # Fit a linear model to the data.
    res = minimize(fun, (0.95, np.mean(t), 1.0), bounds=bounds)
    
    p, alpha, beta = res.x
    
    # Output calculated parameters
    return p, alpha, beta
def gumbel_lfp_lsq(f, t, censored=None, plot=False):
    # Fits a Weibull model using cumulative probability and times (or stress) to failure.
    if len(f) == 0 or len(t) == 0:
        return None
    if censored is None:
        censored = np.zeros(len(t))
        
    t = t[~np.isnan(f)]
    f = f[~np.isnan(f)]
    
    # Create anonymous function to use with optimise
    fun = lambda x : sum(((x[0] * (np.exp(-np.exp(-(t-x[1])/x[2])))) - f)**2)
    
    # Set bounds for p, alpha, beta
    bounds = ((0, 1), (None, None), (0, None))
    
    # Fit a linear model to the data.
    res = minimize(fun, (0.5, np.mean(t), np.std(t)), bounds=bounds)
    
    p, mu, beta = res.x

    # Optional plot display
    if plot:
        plt.scatter(t, -np.log(-np.log(f)), color='k', marker='+', s=100)
        plt.xlim(min(t)*0.99, max(t)*1.01)
        tt = np.linspace(min(t), max(t), num=100)
        plt.plot(tt, -np.log(-np.log(p*(np.exp(-np.exp(-(tt-mu)/beta))))), color='blue')
    
    # Output calculated parameters
    return p, mu, beta
def gumbel_lsq(f, t, censored=None, plot=False):
    if len(f) == 0 or len(t) == 0:
        return None
    if censored is None:
        censored = np.zeros(len(t))
        
    t = t[~np.isnan(f)]
    f = f[~np.isnan(f)]
    
    x = t
    y = -np.log(-np.log(f))
    
    beta, mu = np.polyfit(x, y, 1)
    model = beta, mu
    beta = 1 / beta
    mu = -mu * beta

    if plot:
        plt.scatter(x, y, color='k', marker='+', s=100)
        plt.xlim(min(x)*0.99, max(x)*1.01)
        t = np.linspace(min(x), max(x), num=100)
        plt.plot(t, model[0] * t + model[1], color='blue')
    
    return mu, beta

class Parametric():
    '''
    This is a class used to create an object that can be used to perform a variety 
    of Reliability Engineering tasks.

    The intent is to encapsulate some reliability functions to reduce the complexity.
    Need to have data for:
    CDF - F
    Rel - R
    PDF - f
    Haz - h
    Cumh- H
    dR  - Reliability Confidence bound.
    given x and c
    
    Model - KM, NA, or FH
    
    
    Need to have functions for:
    Estimate F, R, f, h, H for any x
    
    Confidence Bound Funciton
    
    '''
    def __init__(self):
        # Need:
        # F
        # R
        # h
        # H
        # f
        # dR
        self.R = None
        self.F = None
        self.h = None
        self.H = None
        self.f = None
        
        self.d = None
        self.N = None
        self.c = None
        self.x = None
        
        
        self.model = None
        self._dR = None  
    def __str__(self):
        # Used to automate print(NonParametric()) call
        return "%s Parametric Reliability Model" % self.model
    def confidence_bounds(self, alpha=0.05):
        if self._dR is None:
            self._dR = qt.ppf(alpha/2, self.r[0] - 1)*self._ddR
        return (self.R + self._dR), (self.R - self._dR)
    def upper_confidence_bound(self, confidence=0.95, alpha=None):
        if alpha is None:
            alpha = 1 - confidence
        return self.R - qt.ppf(alpha/2, self.n[0] - 1)* self._ddR  
    def lower_confidence_bound(self, confidence=0.95, alpha=None):
        if alpha is None:
            alpha = 1 - confidence
        return self.R + qt.ppf(alpha/2, self.n[0] - 1)* self._ddR
    def plot(self, confidence_bounds=False):
        R = np.append(1, self.R)
        x = np.append(0, self.x)
        plt.step(x, R, where='post')
        plt.ylim(0, 1)
        plt.xlim(0, max(x))
        plt.show()
    def to_dataframe(self):
        data = {
            'x' : self.x,
            'Failures' : self.d,
            'Failure Prob' : self.F,
            'Reliability' : self.R,
            'Censored' : self.c,
            'Risk Set' : self.r,
            'Hazard Rate' : self.h,
            'Cumulative Haz' : self.H
        }    
        return pd.DataFrame(data)

## Distributions

# Weibull

# Gumbel

# Gamma

# Exponential