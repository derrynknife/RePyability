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
            self.model_parameterization = "non_parametric"
        else:
            raise ValueError("Unknown TTF Model")

        self.cost_rate = np.vectorize(self._cost_rate)
        self.cost_rate_single_cycle = np.vectorize(
            self._cost_rate_single_cycle
        )

    def set_costs_planned_and_unplanned(self, cp, cu):
        assert cp < cu
        self.cp = cp
        self.cu = cu

    def avg_replacement_time(self, t):
        out = quad(self.dist.sf, 0, t)
        return out[0]

    def avg_replacement_time_non_parametric(self, t, interp):
        def func(x):
            return self.dist.sf(x, interp=interp)

        out = quad(func, 0, t, limit=200)
        return out[0]

    def _cost_rate(self, t):
        planned_costs = self.dist.sf(t) * self.cp
        unplanned_costs = (1 - self.dist.sf(t)) * self.cu
        avg_repl_time = self.avg_replacement_time(t)
        return (planned_costs + unplanned_costs) / avg_repl_time

    def _cost_rate_single_cycle(self, t):
        planned_costs = self.dist.sf(t) * self.cp / t

        def f(x):
            return self.dist.df(x) / x

        unplanned_costs = self.cu * quad(f, 0, t)[0]
        return planned_costs + unplanned_costs

    def _log_cost_rate(self, t):
        return np.log(self._cost_rate(t))

    def _log_cost_rate_single_cycle(self, t):
        return np.log(self._cost_rate_single_cycle(t))

    def _cost_rate_non_parametric(self, t, interp):
        R = self.dist.sf(t, interp=interp)
        planned_costs = R * self.cp
        unplanned_costs = (1 - R) * self.cu

        avg_repl_time = self.avg_replacement_time_non_parametric(t, interp)
        return (planned_costs + unplanned_costs) / avg_repl_time

    def _log_cost_rate_non_parametric(self, t, interp):
        return np.log(self._cost_rate_non_parametric(t, interp=interp))

    def find_optimal_replacement(self, interp=None):
        if self.model_parameterization == "parametric":
            if self.dist.dist.name == "Weibull":
                if self.dist.params[1] <= 1:
                    return np.inf

            init = self.dist.mean()
            bounds = ((1e-8, None),)
            res = minimize(self._log_cost_rate, init, bounds=bounds, tol=1e-10)
            self.optimisation_results = res
            return res.x[0]
        else:
            if interp is None:
                raise ValueError(
                    "When using Non-Parametric model must select \
                    interpolation method."
                )
            elif interp == "step":
                x = self.dist.x
                RUL = (np.diff(x, prepend=0) * self.dist.sf(x)).cumsum()
                cput = (
                    self.cp * self.dist.sf(x) + self.cu * self.dist.ff(x)
                ) / RUL
                self.cput = cput
                return x[np.argmin(cput)]
            elif interp == "linear":
                init = self.dist.x.mean()
                bounds = ((self.dist.x.min(), self.dist.x.max()),)

                def func(x):
                    return self._cost_rate_non_parametric(x, interp)

                res = minimize(func, init, bounds=bounds, tol=1e-10)
                self.optimisation_results = res
                return res.x[0]
            else:
                raise ValueError("`interp` must be either 'step' or 'linear'")

        return res.x[0]

    def find_optimal_replacement_single_cycle(self):
        if self.dist.dist.name == "Weibull":
            if self.dist.params[1] <= 1:
                return np.inf
        init = self.dist.mean()
        bounds = ((1e-8, None),)
        res = minimize(
            self._log_cost_rate_single_cycle, init, bounds=bounds, tol=1e-10
        )
        self.optimisation_results = res
        return res.x[0]
