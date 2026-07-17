import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from surpyval import ExactEventTime, NonParametric, Parametric

from repyability.rbd._model_utils import distribution_name, model_mean
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
        if isinstance(reliability, Parametric):
            self.model_parameterization = "parametric"
            self.reliability_function = reliability.sf
        elif isinstance(reliability, NonParametric):
            # TODO: Allow non-interpolated?
            self.model_parameterization = "non-parametric"
            self.reliability_function = interp1d(
                reliability.x, 1 - reliability.F, fill_value="extrapolate"
            )
        elif isinstance(reliability, StandbyModel):
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
        # surpyval >= 0.11 returns 1-element arrays from sf/mean/qf, and t can
        # arrive as a 1-element array from scipy.optimize.minimize. Coerce to a
        # plain float so the quadrature bound in avg_replacement_time() is a
        # scalar (scipy.integrate.quad rejects array bounds).
        t = float(np.asarray(t).item())
        planned_costs = self.reliability_function(t) * self.cp
        unplanned_costs = (1 - self.reliability_function(t)) * self.cu
        avg_repl_time = self.avg_replacement_time(t)
        return (planned_costs + unplanned_costs) / avg_repl_time

    def _log_cost_rate(self, t):
        return np.log(self._cost_rate(t))

    def mean_unavailability(self) -> float:
        return 1 - self.mean_availability()

    def mean_availability(self) -> float:
        """Long-run availability, MTTF / (MTTF + MTTR)."""
        if isinstance(self.reliability, NonParametric):
            raise ValueError(
                "Mean Availability requires a parametric reliability model"
            )
        mttf = model_mean(self.reliability)
        mttr = model_mean(self.time_to_replace)
        return mttf / (mttr + mttf)

    def failure_frequency(self) -> float:
        """Long-run failure frequency (failures per unit time).

        For an alternating renewal process (fail, repair, fail, ...) this is
        ``1 / (MTTF + MTTR)`` — one failure per mean up-down cycle.
        """
        if isinstance(self.reliability, NonParametric):
            raise ValueError(
                "Failure frequency requires a parametric reliability model"
            )
        mttf = model_mean(self.reliability)
        mttr = model_mean(self.time_to_replace)
        return 1.0 / (mttf + mttr)

    def _cost_rate_with_log_x(self, x):
        return self._cost_rate(np.exp(x))

    def _log_cost_rate_with_log_x(self, x):
        return self._cost_rate(np.exp(x))

    def find_optimal_replacement(self, options=None):
        if self.model_parameterization == "parametric":
            if distribution_name(self.reliability) == "Weibull":
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
