from types import SimpleNamespace

from datenwissenschaften.rnd.model import AdaptiveRecurrentRNDPPO


def test_attaching_rnd_restores_the_models_adaptation_multiplier():
    applied = []
    model = object.__new__(AdaptiveRecurrentRNDPPO)
    model.rnd = SimpleNamespace(set_adaptation_multiplier=applied.append)
    model.adaptation_multiplier = 3.25
    model.env = None

    model._attach_rnd_to_env()

    assert applied == [3.25]
