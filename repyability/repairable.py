from scipy.optimize import minimize


class Repairable:
    """
    Class to store the non-repairable information
    """

    def __init__(self, distribution):
        self.dist = distribution

    def set_repair_and_overhaul_costs(self, cr, co):
        if cr >= co:
            raise ValueError(
                "repair cost, cr, must be less than overhaul cost, co."
            )
        self.cr = cr
        self.co = co

    def cost(self, t):
        return self.cr * self.dist.cif(t) + self.co

    def cost_rate(self, t):
        return self.cost(t) / t

    def find_optimal_overhaul_interval(self):

        return minimize(self.cost_rate, 1.0)
