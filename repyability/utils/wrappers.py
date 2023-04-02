def check_probability(func):
    """checks probability is between 0 and 1"""

    def wrap(obj, target: float, *args, **kwargs):
        if target > 1:
            raise ValueError("target cannot be above 1.")
        elif target < 0:
            raise ValueError("target cannot be below 0.")
        else:
            return func(obj, target, *args, **kwargs)

    return wrap
