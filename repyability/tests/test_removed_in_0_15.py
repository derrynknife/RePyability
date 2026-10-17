"""What 0.14 deprecated is gone in 0.15: ``NodeState``'s ``phase``,
renamed ``since_scheduled`` (#350), as a phased mission has phases of its
own."""

import pytest

from repyability import NodeState


def test_phase_is_refused_for_since_scheduled():
    with pytest.raises(TypeError, match="phase"):
        NodeState(age=10.0, phase=4.0)
    state = NodeState(age=10.0, since_scheduled=4.0)
    assert state.since_scheduled == 4.0
    assert not hasattr(state, "phase")
