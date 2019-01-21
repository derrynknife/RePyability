import scipy
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm

class NonRepairable():

    def __init__(self, distribution):
        self.dist = distribution
        self.cost_rate = np.vectorize(self._cost_rate)

    def set_costs_planned_and_unplanned(self, cp, cu):
        self.cp = cp
        self.cu = cu

    def avg_replacement_time(self, t):
        return quad(self.dist.sf, 0, t)[0]

    def _cost_rate(self, t):
        planned_costs = self.dist.sf(t) * self.cp
        unplanned_costs = (1 - self.dist.sf(t)) * self.cu
        avg_repl_time = self.avg_replacement_time(t)
        return (planned_costs + unplanned_costs) / avg_repl_time

    def find_optimal_replacement(self):
        init = self.dist.mean()
        bounds = ((0, None),)
        res = minimize(self.cost_rate, init, bounds=bounds)
        self.optimisation_results = res
        return res.x