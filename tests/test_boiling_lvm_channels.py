"""Strict channel preservation refuses incomplete or ambiguous LVM records."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/physics_engine/electrochem'))
from boiling_lvm_channels_v1 import load_channels,LVMContractError


def fixture():
    return '\n'.join(['Separator\tTab','Decimal_Separator\t.','Multi_Headings\tNo','Channels\t6',
                       'Y_Unit_Label\t'+'\t'.join(['Deg C']*6),
                       'X_Value\t'+'\t'.join('Temperature_'+str(i) for i in range(6))+'\tComment',
                       '0\t1\t2\t3\t4\t5\t6','1\t7\t8\t9\t10\t11\t12'])+'\n'


def test_all_channels_preserved(tmp_path):
    p=tmp_path/'input.lvm';p.write_text(fixture());a,m=load_channels(p)
    assert a.shape==(2,7) and np.array_equal(a[:,6],[6.,12.])
    assert m['columns'][-1]=='Temperature_5'


@pytest.mark.parametrize('mutation',('channel_count','missing_channel','labels','nan','time','unit'))
def test_bad_layout_refused(tmp_path,mutation):
    s=fixture()
    if mutation=='channel_count':s=s.replace('Channels\t6','Channels\t5')
    if mutation=='missing_channel':s=s.replace('0\t1\t2\t3\t4\t5\t6','0\t1\t2\t3\t4\t5')
    if mutation=='labels':s=s.replace('Temperature_5','Temperature_6')
    if mutation=='nan':s=s.replace('11\t12','11\tnan')
    if mutation=='time':s=s.replace('1\t7\t8','0\t7\t8')
    if mutation=='unit':s=s.replace('Deg C','Kelvin')
    p=tmp_path/'input.lvm';p.write_text(s)
    with pytest.raises(LVMContractError):load_channels(p)
