import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from surpyval.nonparametric import NonParametric
from surpyval.parametric import Parametric
from surpyval.parametric.exact_event_time import ExactEventTime


class NonRepairable:
    """
    Class to store the non-repairable information
    """

    def __init__(
        self, reliability, time_to_replace=ExactEventTime.from_params(0)
    ):
        if type(reliability) == Parametric:
            self.model_parameterization = "parametric"
        elif type(reliability) == NonParametric:
            raise NotImplementedError("Non-Parametric Models not implemented")
        else:
            raise ValueError("Unknown TTF Model")

        self.reliability = reliability
        self.time_to_replace = time_to_replace
        self.cost_rate = np.vectorize(self._cost_rate)
        self.__next_event_type = "failure"

    def set_costs_planned_and_unplanned(self, cp, cu):
        assert cp < cu
        self.cp = cp
        self.cu = cu

    def avg_replacement_time(self, t):
        out = quad(self.reliability.sf, 0, t)
        return out[0]

    def _cost_rate(self, t):
        planned_costs = self.reliability.sf(t) * self.cp
        unplanned_costs = (1 - self.reliability.sf(t)) * self.cu
        avg_repl_time = self.avg_replacement_time(t)
        return (planned_costs + unplanned_costs) / avg_repl_time

    def _log_cost_rate(self, t):
        return np.log(self._cost_rate(t))

    def find_optimal_replacement(self, interp=None):
        if self.reliability.dist.name == "Weibull":
            if self.reliability.params[1] <= 1:
                return np.inf

        init = self.reliability.mean()
        bounds = ((1e-8, None),)
        res = minimize(self._log_cost_rate, init, bounds=bounds, tol=1e-10)
        self.optimisation_results = res
        return res.x[0]

    def next_event(self):
        if self.__next_event_type == "failure":
            self.__next_event_type = "replace"
            return self.reliability.random(1).item(), False
        elif self.__next_event_type == "replace":
            self.__next_event_type = "failure"
            return self.time_to_replace.random(1).item(), True
