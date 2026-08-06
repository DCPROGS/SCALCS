"""SCALCS can load a mechanism from the modern dcio JSON format.

Round-trip integrity: a mechanism loaded directly from a legacy .mec file
must be reproduced when that mechanism is serialised to JSON by
``dcio.formats.modern`` and read back through ``scalcsio.mec_load_json``.
"""

from pathlib import Path

import numpy as np
import pytest

dcio_mec = pytest.importorskip("dcio.formats.mec")
dcio_modern = pytest.importorskip("dcio.formats.modern")

from scalcs import scalcsio

_QMECHDEM = Path(r"E:\dcprogs\dcdos\qmechdem.mec")


def _need(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"{p} not found")
    return p


@pytest.fixture(scope="module")
def mec_path():
    return _need(_QMECHDEM)


@pytest.fixture(scope="module")
def direct_mec(mec_path):
    """Mechanism loaded straight from the .mec file (Milone model 29)."""
    _, entries, _ = dcio_mec.read_list(mec_path)
    start = next(e.start_byte for e in reversed(entries) if e.mec_num == 29)
    return scalcsio.mec_load(mec_path, start)


@pytest.fixture(scope="module")
def json_mec(mec_path, tmp_path_factory):
    """Same mechanism, serialised to JSON and loaded back via SCALCS."""
    _, entries, _ = dcio_mec.read_list(mec_path)
    entry = next(e for e in reversed(entries) if e.mec_num == 29)
    rec = dcio_mec.read(mec_path, entry)
    out = tmp_path_factory.mktemp("modern") / "milone.json"
    dcio_modern.write_mechanism(rec, out)
    return scalcsio.mec_load_json(out)


class TestMechanismFromJson:
    def test_returns_mechanism(self, json_mec):
        from scalcs import mechanism
        assert isinstance(json_mec, mechanism.Mechanism)

    def test_state_counts(self, json_mec, direct_mec):
        assert len(json_mec.States) == len(direct_mec.States)
        assert json_mec.kA == direct_mec.kA

    def test_state_names_and_types(self, json_mec, direct_mec):
        assert [s.name for s in json_mec.States] == [
            s.name for s in direct_mec.States
        ]
        assert [s.statetype for s in json_mec.States] == [
            s.statetype for s in direct_mec.States
        ]

    def test_conductances(self, json_mec, direct_mec):
        np.testing.assert_allclose(
            [s.conductance for s in json_mec.States],
            [s.conductance for s in direct_mec.States],
        )

    def test_rate_count_and_names(self, json_mec, direct_mec):
        assert len(json_mec.Rates) == len(direct_mec.Rates)
        assert {r.name for r in json_mec.Rates} == {
            r.name for r in direct_mec.Rates
        }

    def test_cycles(self, json_mec, direct_mec):
        assert len(json_mec.Cycles) == len(direct_mec.Cycles)
        if direct_mec.Cycles:
            assert json_mec.Cycles[0].states == direct_mec.Cycles[0].states

    def test_Q_matrix_matches(self, json_mec, direct_mec):
        for m in (json_mec, direct_mec):
            m.set_eff("c", 100e-9)
        np.testing.assert_allclose(json_mec.Q, direct_mec.Q, rtol=1e-9, atol=0)


class TestErrors:
    def test_non_mechanism_dict_raises(self, tmp_path):
        import json

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"dcio_schema": "config/1.0"}))
        with pytest.raises(ValueError):
            scalcsio.mec_load_json(bad)
