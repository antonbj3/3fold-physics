"""Graph schema rejection and physical-seam null/sensitivity controls."""
import copy
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'src/physics_engine/chains'))
from chain_spec_v1 import execute, generate, validate


def spec(name='litho_from_spec'):
    return json.loads((ROOT/'src/physics_engine/chains/specs'/(name+'.json')).read_text())


@pytest.mark.parametrize('change',('operation','cycle','duplicate','missing_parameter','unknown_override'))
def test_malformed_graph_refuses(change):
    s=spec()
    if change=='operation':s['nodes'][0]['op']='os.system'
    if change=='cycle':s['nodes'][0]['kwargs']['clear']={'ref':'summary.line'}
    if change=='duplicate':s['nodes'][1]['id']='mask'
    if change=='missing_parameter':s['nodes'][0]['kwargs']['clear']={'param':'missing'}
    with pytest.raises(ValueError):
        if change=='unknown_override':execute(s,{'missing':1})
        else:validate(s)


def test_source_generation_ignores_mapping_insertion_order():
    s=spec()
    reordered=json.loads(json.dumps(s,sort_keys=True))
    assert generate(s)==generate(reordered)
    assert b'write_bundle(SPEC)' in generate(s)


def test_clear_mask_terminates_through_all_pore_seams():
    r=execute(spec('litho_pore_from_spec'),{'clear_mask':True})
    assert r['output']=={'line':False}
    assert r['trace']['etched']==r['trace']['pores']


def test_dose_perturbation_changes_resist_and_etch_but_preserves_image():
    s=spec();a=execute(s);b=execute(s,{'dose':60.})
    assert a['trace']['image']==b['trace']['image']
    assert a['trace']['resist']!=b['trace']['resist']
    assert a['trace']['etch']!=b['trace']['etch']
    assert b['output']['cd_etch']<a['output']['cd_etch']
