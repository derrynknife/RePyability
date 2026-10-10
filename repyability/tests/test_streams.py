"""The random streams' uniforms are counter-based (#209): uniform ``k`` of
simulation ``r`` is numpy's Philox's ``k``-th with the stream's key and
the counter ``(0, r, 0, 0)``, whichever function works it out, and the
layout the draws are worked out in (a block's width, its chunks' rows)
changes nothing a run gives."""

import dataclasses

import numpy as np
import pytest

from repyability.rbd import _compiled, _event_loop, _philox, _streams
from repyability.tests.keyed_draws import reference_uniform
from repyability.tests.test_simulation_engines import identical, plain_rbds


def functions():
    yield "numpy", _philox.uniforms
    if _compiled.available():
        from repyability.rbd import _philox_kernel

        yield "compiled", _philox_kernel.uniforms


@pytest.mark.parametrize("name, uniforms", list(functions()))
def test_the_uniforms_are_numpys_philox(name, uniforms):
    rng = np.random.default_rng(3)
    for trial in range(40):
        key = rng.integers(0, 2**63, 2, dtype=np.uint64) * np.uint64(2)
        key += np.uint64(trial % 2)
        sims = rng.integers(0, 2**63, int(rng.integers(1, 5)), np.uint64)
        if trial == 0:
            sims[0] = np.uint64(2**64 - 1)  # the largest counter word
        first, rows = int(rng.integers(0, 13)), int(rng.integers(0, 23))
        got = uniforms(key, sims, first, rows)
        assert got.shape == (sims.size, rows)
        for j, r in enumerate(sims.tolist()):
            counter = np.array([0, r, 0, 0], dtype=np.uint64)
            philox = np.random.Philox(key=key, counter=counter)
            want = np.random.Generator(philox).random(first + rows)[first:]
            assert np.array_equal(got[j].view(np.uint64), want.view(np.uint64))


def test_a_block_holds_the_defined_uniforms_whatever_its_width():
    rbd = plain_rbds()["bridge"]
    plan, _ = _event_loop._stream_plan(rbd, 400.0, 5, False)
    spec = plan.specs[(("a",), _streams.FAILURE)]
    uniform = dataclasses.replace(spec, sampler=lambda u: u)
    for width in (1, 3, 8):
        for antithetic in (False, True):
            wide = dataclasses.replace(uniform, width=width)
            block = _streams.Block(_streams.key(5, wide), wide, 2, antithetic)
            block.extend()
            columns = block.values.shape[0]
            for j in range(columns):
                r = 2 * width + (j // 2 if antithetic else j)
                for k in (0, block.values.shape[1] - 1):
                    u = reference_uniform(5, spec, r, k)
                    if antithetic and j % 2:
                        u = 1.0 - u
                    assert block.values[j, k] == u


@pytest.mark.parametrize("engine", ["python", "numba"])
def test_the_layout_changes_nothing(monkeypatch, engine):
    if engine == "numba":
        pytest.importorskip("numba")
    rbd = plain_rbds()["koon"]
    run = dict(mc_samples=150, seed=4, engine=engine, control_variate=False)
    expected = rbd.availability(300.0, **run)
    monkeypatch.setattr(_streams, "first_rows", lambda expected: 3)
    monkeypatch.setattr(_streams, "BLOCK_DRAWS", 7)
    monkeypatch.setattr(_streams, "MAX_WIDTH", 2)
    identical(expected, rbd.availability(300.0, **run))
    paired = dict(run, antithetic=True)
    monkeypatch.undo()
    expected = rbd.availability(300.0, **paired)
    monkeypatch.setattr(_streams, "block_width", lambda rows: 1)
    identical(expected, rbd.availability(300.0, **paired))
