import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

class NonRepairable():
    """
    Class to store the non-repairable information
    """
    def __init__(self, distribution):
        self.dist = distribution
        self.cost_rate = np.vectorize(self._cost_rate)
        self.cost_rate_single_cycle = np.vectorize(self._cost_rate_single_cycle)

    def set_costs_planned_and_unplanned(self, cp, cu):
        self.cp = cp
        self.cu = cu

    def avg_replacement_time(self, t):
        out = quad(self.dist.sf, 0, t)
        return out[0]

    def _cost_rate(self, t):
        planned_costs = self.dist.sf(t) * self.cp
        unplanned_costs = (1 - self.dist.sf(t)) * self.cu
        avg_repl_time = self.avg_replacement_time(t)
        return (planned_costs + unplanned_costs) / avg_repl_time

    def _cost_rate_single_cycle(self, t):
        planned_costs = self.dist.sf(t) * self.cp / t
        f = lambda x : self.dist.df(x) / x
        unplanned_costs = self.cu * quad(f, 0, t)[0]
        return planned_costs + unplanned_costs

    def _log_cost_rate(self, t):
        return np.log(self._cost_rate(t))
    
    def _log_cost_rate_single_cycle(self, t):
        return np.log(self._cost_rate_single_cycle(t))

    def find_optimal_replacement(self):
        if self.dist.dist.name == "Weibull":
            if self.dist.beta <= 1:
                return np.inf
        init = self.dist.mean()
        bounds = ((1e-8, None),)
        res = minimize(self._log_cost_rate, init, bounds=bounds, tol=1e-10)
        self.optimisation_results = res
        return res.x[0]

    def find_optimal_replacement_single_cycle(self):
        if self.dist.dist.name == "Weibull":
            if self.dist.beta <= 1:
                return np.inf
            else:
                return self.dist.alpha * (self.cp/(self.dist.beta * (self.cu - self.cp))) ** (1./self.dist.beta)
        init = self.dist.mean()
        bounds = ((1e-8, None),)
        res = minimize(self._log_cost_rate_single_cycle, init, bounds=bounds, tol=1e-10)
        self.optimisation_results = res
        return res.x[0]