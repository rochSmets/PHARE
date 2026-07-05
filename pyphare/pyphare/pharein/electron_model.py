from . import global_vars


class IsothermalClosure(object):
    closure_name = "isothermal"

    def __init__(self, **kwargs):
        self.Te = kwargs.get("Te", IsothermalClosure._defaultTe())

    @staticmethod
    def _defaultTe():
        return 0.1

    def dict_path(self):
        return {"name/": IsothermalClosure.closure_name, "Te": self.Te}

    @staticmethod
    def name():
        return IsothermalClosure.closure_name


class ElectronModel(object):
    def __init__(self, **kwargs):
        if kwargs["closure"] == "isothermal":
            self.closure = IsothermalClosure(**kwargs)
        else:
            self.closure = None

        self.min_density = kwargs.get("min_density", 0.0)
        if self.min_density < 0.0:
            raise ValueError("Error: min_density should not be negative")

        global_vars.sim.set_electrons(self)

    def dict_path(self):
        return [
            ("electrons/pressure_closure/" + k, v)
            for k, v in self.closure.dict_path().items()
        ] + [("electrons/min_density", self.min_density)]
