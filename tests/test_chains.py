"""Pytest wrapper for the coupled chains under src/physics_engine/chains/: the chain is run as a script in a fresh
subprocess with src/physics_engine as the working directory and must exit 0, and its three composition checks
(uncertainty propagation, cert composition, null cases) must all print PASS.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "physics_engine")
ENV = dict(os.environ, MPLBACKEND="Agg")


def test_litho_chain_v1():
    proc = subprocess.run([sys.executable, os.path.join(SRC, "chains", "litho_chain_v1.py")],
                          cwd=SRC, env=ENV, capture_output=True, text=True, timeout=600)
    out = proc.stdout
    assert proc.returncode == 0, out[-4000:] + proc.stderr[-4000:]
    assert "FAIL" not in out, out[-4000:]
    for needle in ("uncertainty propagation", "cert composition", "null cases", "CHAIN RESULT"):
        assert needle in out, out[-4000:]
