import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from surpyval.nonparametric import NonParametric
from surpyval.parametric import Parametric


class NonRepairable:
    """
    Class to store the non-repairable information
    """

    def __init__(self, distribution):
        self.dist = distribution
        if type(distribution) == Parametric:
            self.model_parameterization = "parametric"
        elif type(distribution) == NonParametric:
            raise NotImplementedError("Non-Parametric Models not implemented")
        else:
            raise ValueError("Unknown TTF Model")

        self.cost_rate = np.vectorize(self._cost_rate)

    def set_costs_planned_and_unplanned(self, cp, cu):
        assert cp < cu
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

    def _log_cost_rate(self, t):
        return np.log(self._cost_rate(t))

    def find_optimal_replacement(self, interp=None):
        if self.dist.dist.name == "Weibull":
            if self.dist.params[1] <= 1:
                return np.inf

        init = self.dist.mean()
        bounds = ((1e-8, None),)
        res = minimize(self._log_cost_rate, init, bounds=bounds, tol=1e-10)
        self.optimisation_results = res
        return res.x[0]
