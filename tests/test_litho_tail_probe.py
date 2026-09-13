"""Pore observer null and tail-direction regression controls."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/physics_engine/chains'))
from litho_tail_probe_v1 import pore, tail


def test_rare_losses_widen_tail_and_zero_probability_is_identity():
    widths=np.full(10000,45.)
    draws=np.ones(10000);draws[:200]=0.
    radii=np.full(10000,5.)
    assert np.array_equal(pore(widths,draws,radii,probability=0.),widths)
    assert tail(pore(widths,draws,radii))==10.
    assert tail(widths)==0.
    assert np.all(pore(widths,draws,radii*100.)>=0.)
