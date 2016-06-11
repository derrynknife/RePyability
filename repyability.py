import pandas as pd
import re
import matplotlib.pyplot as plt
import brewer2mpl
import scipy
import numpy as np
import nltk
from collections import Counter
from datetime import datetime, date, time, timedelta
from scipy.stats import t as qt
from scipy.optimize import minimize

## Non-Repairable Components

def plotting_positions(t, censored=None, formula="Blom"):
    # Numbers from "Effect of Renking Selection on the Weibull Modulus Estimation"
    # Authors: Kirtay, S; Dispinar, D.
    # From: Gazi University Journal of Science 25(1):175-187, 2012.
    # Assumes no repeated times.

    # Adust ranks if censored data present
    if censored is not None:
        ranks = rank_adjust(t, censored)
    else:
        ranks = np.array(range(0, len(t))) + 1

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
    elif formula == "Tukey":      A, B = 1.0/3.0, 1.0/3.0
    elif formula == "DPW":        A, B = 1.0, 0.0

    # Use general adjustment formula
    pp = (ranks - A)/(len(ranks) + B)
    return pp

def rank_adjust(t, censored=None):
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
    for i in range(0,n):
        if censored[i] == 0:
            ranks[i] = PMON + (n + 1 - PMON)/(n - i + 2)
            PMON = ranks[i]
        else:
            ranks[i] = np.nan
    # Return adjusted ranks
    return ranks

def weibull_lsq(f, t, censored=None, plot=False):
    # Fits a Weibull model using cumulative probability and times (or stress) to failure.
    if len(f) == 0 or len(t) == 0:
        return None
    if censored is None:
        censored = np.zeros(len(t))

    # Convert data to linearised form
    x = np.log(t[censored == 0])
    y = np.log(np.log(1/(1 - f[censored == 0])))
    
    # Fit a linear model to the data.
    model = np.polyfit(x, y, 1)
    
    # Compute alpha and beta from linearised least square model
    beta  = model[0]
    alpha = np.exp(model[1]/-beta)

    # Optional plot display
    if plot:
        plt.scatter(x, y, color='k', marker='+', s=100)
        plt.xlim(min(x)*0.99, max(x)*1.01)
        t = np.linspace(min(x), max(x), num=100)
        plt.plot(t, model[0] * t + model[1], color='blue')
    
    # Output calculated parameters
    return alpha, beta

def weibull_lfp_lsq(f, t, censored=None, plot=False):
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

    # Optional plot display
    if plot:
        plt.scatter(np.log(t), np.log(np.log(1/(1-f))), color='k', marker='+', s=100)
        plt.xlim(min(np.log(t))*0.99, max(np.log(t))*1.01)
        tt = np.linspace(min(np.log(t)), max(np.log(t)), num=100)
        plt.plot(tt, np.log(np.log(1/(1-(p * (1 - np.exp(-(np.exp(tt)/alpha)**beta)))))), color='blue')
    
    # Output calculated parameters
    return p, alpha, beta

def weibull_mle(t, censored=None, plot=False):
    # Fits a Weibull model using cumulative probability and times (or stress) to failure.
    t = sorted(t)
    if len(t) == 0:
        return None
    if censored is None:
        censored = np.zeros(len(t))
        
    # Create anonymous function to use with optimise
    fun = lambda x: -sum((1-censored)*(np.log(x[1]/x[0]) + 
        (x[1]-1)*np.log(t/x[0]) - (t/x[0])**(x[1])) - (censored)*((t/x[0])**(x[1])))
    
    # Set bounds for alpha, beta
    bounds = ((0, None), (0, None))
    
    # Fit a linear model to the data.
    res = minimize(fun, (np.mean(t), 1.0), bounds=bounds)

    alpha, beta = res.x

    # Optional plot display
    if plot:
        f = 1 - nelson_aalen(t, censored)
        plt.scatter(np.log((t[~np.isnan(f)])), np.log(np.log(1/(1-f))), color='k', marker='+', s=100)
        plt.xlim(min(np.log(t))*0.99, max(np.log(t))*1.01)
        tt = np.linspace(min(np.log(t)), max(np.log(t)), num=100)
        plt.plot(tt, np.log(np.log(1/(1-((1 - np.exp(-(np.exp(tt)/alpha)**beta)))))), color='blue')
    # Output calculated parameters
    return alpha, beta

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

def nelson_aalen(t, censored=None, plot=False, cb=False, alpha=0.05):
    # Nelson-Aalen estimation of Reliability function
    # Nelson, W.: Theory and Applications of Hazard Plotting for Censored Failure Data. 
    # Technometrics, Vol. 14, #4, 1972
    # Technically the NA estimate is for the Cumulative Hazard Function,
    # The reliability (survival) curve that is output is also known as the Breslow estimate.
    # I will leave it as Nelson-Aalen for this library.

    # Assumes whole population given
    n = len(t)
    nn = len(set(t))

    # Rank computations for:
    # Items at risk
    # ni = n(i-1) - d(i-1) - c(i-1)
    # Hazard Rate
    # hi = di/ni
    # Cumulative Hazard Function
    # Hi = cumsum(hi)
    # Reliability Function
    # Ri = exp(-Hi)

    if censored is None:
        # Uncensored data
        t, d = np.unique(t, return_counts=True)
        ni = n - np.cumsum(d) + np.cumsum(d)[0]
        ni = [float(x) for x in ni]
        hi = (d/ni)
        times = t
    else:
        counts = {}
        for x, c in zip(t, censored):
            a = np.zeros(2)
            f = counts.get(x, a)
            f[c] = 1 + f[c]
            counts[x] = f
        
        times = np.array(sorted(counts.keys()))
        r = np.array([counts[x][1] for x in times])
        d = np.array([counts[x][0] for x in times])
        ni = n - np.cumsum(d) - np.cumsum(r) + np.cumsum(d)[0] + np.cumsum(r)[0]
        ni = [float(x) for x in ni]
        hi = (d/ni)
    
    Hi = np.cumsum(hi)
    Ri = np.exp(-Hi)
    
    R_ll = Ri + qt.ppf(alpha/2, n - 1)*np.sqrt(Ri**2 * np.cumsum((d)/(ni*(ni-(d)))))
    R_ul = Ri - qt.ppf(alpha/2, n - 1)*np.sqrt(Ri**2 * np.cumsum((d)/(ni*(ni-(d)))))

    if plot:
        plt.step(times, Ri, color='k')
        if cb:
            plt.step(times, R_ll, color='r')
            plt.step(times, R_ul, color='r')
    
    return Ri

def kaplan_meier(t, censored=None, plot=False, cb=False, alpha=0.05):
    # Kaplan-Meier estimate of survival
    # Good explanation of K-M reason can be found at:
    # http://reliawiki.org/index.php/Non-Parametric_Life_Data_Analysis#Kaplan-Meier_Example
    # Data given not necessarily in order
    # Assumes whole population given

    n = len(t)

    if censored is None:
        # Uncensored data set
        t, d = np.unique(t, return_counts=True)
        ni = n - np.cumsum(d) + np.cumsum(d)[0]
        # Conditional survival
        ni = [float(x) for x in ni]
        Ri = (ni - d) / ni
        # Nonparametric survival plot   
    else:
        counts = {}
        for x, c in zip(t, censored):
            a = np.zeros(2)
            f = counts.get(x, a)
            f[c] = 1 + f[c]
            counts[x] = f
        
        t = np.array(sorted(counts.keys()))
        r = np.array([counts[x][1] for x in t])
        d = np.array([counts[x][0] for x in t])
        ni = n - np.cumsum(d) - np.cumsum(r) + np.cumsum(d)[0] + np.cumsum(r)[0]
        ni = [float(x) for x in ni]
        Ri = (ni - d)/ni
    
    Ri = np.cumprod(Ri)

    dR = qt.ppf(alpha/2, n - 1)*np.sqrt(Ri**2 * np.cumsum((d)/(ni*(ni-(d)))))
    
    R_ll = Ri + dR
    R_ul = Ri - dR
    
    if plot:
        plt.step(t, Ri, color='k')
        if cb:
            plt.step(t, R_ll, color='r')
            plt.step(t, R_ul, color='r')
    
    return Ri

def fleming_harrington(t, censored=None, plot=False, cb=False, alpha=0.05):
    # Fleming-Harrington estimation of Reliability function (via cum hazard)
    # Fleming, T. R., and Harrington, D. P. (1984). “Nonparametric Estimation of the Survival Distribution in Censored Data.” 
    # Communications in Statistics—Theory and Methods 13:2469–2486.
    # Equation taken from http://support.sas.com/documentation/cdl/en/statug/68162...
    # .../HTML/default/viewer.htm#statug_lifetest_details03.htm

    # Get the at risk total, Y, from t = 0
    Y = len(t)
    
    order = np.argsort(t)
    t = t[order]
    if censored is not None:
        assert len(t) == len(censored)
        c = np.array(censored)[order]

    if censored is None:
        # Uncensored data
        # for each time, t, find number of failures, d.
        t, d = np.unique(t, return_counts=True)
        hi = np.zeros_like(t)
        for i in range(len(t)):
            # For each event time, divide by the surviving set, Y
            # This is the critical difference to Nelson-Aalen
            hi[i] = np.sum([1./(Y-j) for j in range(d[i])])
            Y -= d[i]
        ni = np.cumsum(d)[::-1]
    else:
        # With censored data
        dhi = np.zeros_like(t)
        unique_t = np.unique(t)
        hi = np.zeros_like(unique_t)
        ni = np.zeros_like(unique_t)
        for i in range(len(t)):
            dhi[i] = float(c[i]) / Y
            Y -= 1
        for i in range(len(unique_t)):
            hi[i] = np.sum(dhi[t == unique_t[i]])
            ni[i] = np.sum(c[t == unique_t[i]])
        d  = ni
        ni = np.cumsum(ni)[::-1]
        t = unique_t
    
    Hi = np.cumsum(hi)
    Ri = np.exp(-Hi)
    
    dR = qt.ppf(alpha/2, ni[0] - 1)*np.sqrt(Ri**2 * np.cumsum(d/(ni*(ni-d))))
    
    R_ll = Ri + dR
    R_ul = Ri - dR

    if plot:
        plt.step(t, Ri, color='k')
        if cb:
            plt.step(t, R_ll, color='r')
            plt.step(t, R_ul, color='r')
    
    return Ri


class NonRepairable():
    def __init__():
        self.a = "Hello"