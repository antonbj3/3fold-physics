import numpy as np
import pytest
from physics_engine.process import weld_goldak_spot_nist as spot
from physics_engine.process import weld_goldak_spot_nist_gpu as sg
from physics_engine.process import lpbf_meltpool_render_match as lpbf
from physics_engine.process import lpbf_meltpool_render_match_gpu as lg

def test_power_interpolation_and_finite_time_quadrature():
    args=(np.array([0.,.0001,.001,.003]),np.linspace(0,1e-4,11),np.array([0.,.0001,.002]),np.array([0.,300.,200.]),spot.AL)
    expected=spot.temp_field(*args,nu=61)
    for tile in (1,8):np.testing.assert_allclose(sg.temp_field(*args,nu=61,device='cpu',tile=tile),expected,atol=1e-8,rtol=1e-9)

@pytest.mark.parametrize('beam',[1e-7,3.0625e-5])
def test_spatial_broadcast_quadrature_tiles_and_point_limit(beam):
    args=(np.linspace(-1e-4,0,3)[:,None],np.linspace(0,1e-4,5)[None,:],np.array(1e-7),473.,.7,.433,beam,150.,2700.,1050.)
    expected=lpbf.dT_eagar_tsai(*args,n_u=101)
    for tile in (2,2048):np.testing.assert_allclose(lg.dT_eagar_tsai(*args,n_u=101,device='cpu',tile=tile),expected,atol=1e-8,rtol=1e-9)
    args=args[:7]+args[8:]
    np.testing.assert_allclose(lg.dT_rosenthal(*args,device='cpu'),lpbf.dT_rosenthal(*args),atol=1e-8,rtol=1e-9)
