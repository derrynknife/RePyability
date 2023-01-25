import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from surpyval.nonparametric import NonParametric
from surpyval.parametric import Parametric
from surpyval.parametric.exact_event_time import ExactEventTime

from repyability.rbd.standby_node import StandbyModel

FAILURE = 1
REPLACE = 0


class NonRepairable:
    """
    Class to store the non-repairable information
    """

    def __init__(
        self, reliability, time_to_replace=ExactEventTime.from_params(0)
    ):
        if type(reliability) == Parametric:
            self.model_parameterization = "parametric"
            self.reliability_function = reliability.sf
        elif type(reliability) == NonParametric:
            # TODO: Allow non-interpolated?
            self.model_parameterization = "non-parametric"
            self.reliability_function = interp1d(
                reliability.x, 1 - reliability.F, fill_value="extrapolate"
            )
        elif type(reliability) == StandbyModel:
            self.model_parameterization = "non-parametric"
            self.reliability_function = interp1d(
                reliability.model.x,
                1 - reliability.model.F,
                fill_value="extrapolate",
            )
        else:
            raise ValueError("Unknown reliability function")

        self.reliability = reliability
        self.time_to_replace = time_to_replace
        self.cost_rate = np.vectorize(self._cost_rate)
        self.__next_event_type = FAILURE

    def set_costs_planned_and_unplanned(self, cp, cu):
        if cp >= cu:
            raise ValueError("Planned costs must be less than unplanned costs")
        self.cp = cp
        self.cu = cu

    def avg_replacement_time(self, t):
        if self.model_parameterization == "parametric":
            out = quad(self.reliability_function, 0, t)[0]
        else:
            mask = self.reliability.x < t
            x_less_than_t = self.reliability.x[mask]
            dt = np.diff(x_less_than_t)
            F = self.reliability_function(x_less_than_t[1:])
            out = (dt * F).sum()

        return out

    def _cost_rate(self, t):
        planned_costs = self.reliability_function(t) * self.cp
        unplanned_costs = (1 - self.reliability_function(t)) * self.cu
        avg_repl_time = self.avg_replacement_time(t)
        return (planned_costs + unplanned_costs) / avg_repl_time

    def _log_cost_rate(self, t):
        return np.log(self._cost_rate(t))

    def q(self, t):
        return np.log(self._cost_rate(t))

    def mean_unavailability(self):
        return 1 - self.mean_availability()

    def mean_availability(self):
        if isinstance(self.reliability, NonParametric):
            raise ValueError(
                "Mean Availability requires a parametric reliability model"
            )
        mttf = self.reliability.mean()
        mttr = self.time_to_replace.mean()
        return mttf / (mttr + mttf)

    def _cost_rate_with_log_x(self, x):
        return self._cost_rate(np.exp(x))

    def _log_cost_rate_with_log_x(self, x):
        return self._cost_rate(np.exp(x))

    def find_optimal_replacement(self, options=None):
        if self.model_parameterization == "parametric":
            if self.reliability.dist.name == "Weibull":
                if self.reliability.offset:
                    pass
                elif self.reliability.zi:
                    pass
                elif self.reliability.lfp:
                    pass
                elif self.reliability.params[1] <= 1:
                    return np.inf
            # When using a parametric distribution the optimisation is
            # straight forward. Simply find the point in the support where
            # the cost rate is minimised. Uses quadrature to integrate!
            mean = self.reliability.mean()
            old_err_state = np.seterr(all="ignore")
            res = minimize(self._cost_rate_with_log_x, np.log(mean), tol=1e-10)
            res_log = minimize(
                self._log_cost_rate_with_log_x, np.log(mean), tol=1e-10
            )
            np.seterr(**old_err_state)
            self.optimisation_results = res
            if res["fun"] < np.exp(res_log["fun"]):
                optimal = np.exp(res.x[0])
            else:
                optimal = np.exp(res_log.x[0])
        else:
            # When using non-parametric estimations, it can also be straight
            # forward. Simply find the cost rate at a number of places
            # from the min to the max support of the model and return the
            # value of x which has the minimum cost rate.
            x_search = np.linspace(
                self.reliability.x.min(), self.reliability.x.max(), 10000
            )

            dt = np.diff(x_search, prepend=0)
            R = self.reliability_function(x_search)
            avg_replacement_times = (dt * R).cumsum()

            planned_costs = R * self.cp
            unplanned_costs = (1 - R) * self.cu

            costs = (planned_costs + unplanned_costs) / avg_replacement_times

            optimal_idx = np.argmin(costs)
            optimal = x_search[optimal_idx]
        return optimal

    def reset(self):
        self.__next_event_type = FAILURE

    def next_event(self):
        if self.__next_event_type == FAILURE:
            self.__next_event_type = REPLACE
            return self.reliability.random(1).item(), False
        elif self.__next_event_type == REPLACE:
            self.__next_event_type = FAILURE
            return self.time_to_replace.random(1).item(), True
