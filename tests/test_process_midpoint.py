"""Finite-area sampling and unchanged constitutive response controls."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'src/physics_engine/chains'))
import process_midpoint_v1 as model


def test_cell_centers_integrate_area_and_refine_quadratic_moment():
    errors=[]
    for n in (41,123,369):
        y=model.fiber_centers(n)
        assert np.all(np.abs(y)<.0004)
        assert abs(np.mean(y))<1e-18
        errors.append(abs(np.mean(y*y)-.0004**2/3))
    assert errors[1]<errors[0]/8
    assert errors[2]<errors[1]/8
    assert np.max(np.abs(model.fiber_centers(41)-model.fiber_centers(123)[1::3]))<1e-18


@pytest.mark.parametrize('count',(True,0,2,3.5))
def test_invalid_resolution_refuses(count):
    with pytest.raises(ValueError):model.fiber_centers(count)


def test_frozen_material_response_is_exact_on_supplied_field(monkeypatch):
    field=np.array([[0,0,0,0,0],[0,200,2000,200,0],[0,100,800,100,0],[0,0,0,0,0]],float)
    monkeypatch.setattr(model.frozen,'NFIBER',5)
    expected=model.frozen.run_state(field,cooling=17)
    actual=model.run_state(field,cooling=17)
    for key in ('residual','low','high'):
        assert expected[key].tobytes()==actual[key].tobytes()
    assert expected['hashes']==actual['hashes']
    assert expected['diagnostics']==actual['diagnostics']
    assert all(actual['gates'].values())
