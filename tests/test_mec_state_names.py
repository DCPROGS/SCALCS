"""Tests for reading state names from a .mec file.

Each name is stored in a 10-byte field. Some writers pad it with spaces and
some with NUL bytes, and older files leave it blank, never having stored the
names at all. A blank field falls back to a positional name, S1, S2, ...,
numbered as the states are in the file - which is also how the drawing stored
in the file numbers them.
"""
import os

import pytest

from scalcs import scalcsio


class TestStateNameField:

    @pytest.mark.parametrize('field, name', [
        (b'AR*       ', 'AR*'),                       # padded with spaces
        (b'AR*\x00\x00\x00\x00\x00\x00\x00', 'AR*'),  # padded with NULs
        (b'\x00\x00A2R\x00  \x00\x00', 'A2R'),        # either, either side
        (b'A2R junk  ', 'A2R'),                       # first word only, as before
    ])
    def test_padding_of_either_kind_is_removed(self, field, name):
        assert scalcsio._state_name(field, 3) == name

    def test_a_field_padded_with_nuls_never_gives_a_name_containing_one(self):
        """split() does not count NUL as whitespace, so the old reader named
        such states with ten NUL bytes: the file loaded, but every state of
        it had the same unprintable name."""
        assert '\x00' not in scalcsio._state_name(b'R' + b'\x00' * 9, 1)

    @pytest.mark.parametrize('field', [b' ' * 10, b'\x00' * 10, b' \x00' * 5])
    def test_a_blank_field_gets_the_state_s_number(self, field):
        """Space-padded, the old reader raised IndexError; NUL-padded, it
        returned ten NULs."""
        assert scalcsio._state_name(field, 4) == 'S4'


MEC_DIR = os.environ.get('SCALCS_MEC_DIR', '')


def _load(name):
    path = os.path.join(MEC_DIR, name)
    if not os.path.isfile(path):
        pytest.skip('set SCALCS_MEC_DIR to a folder holding %s' % name)
    _, meclist, _ = scalcsio.mec_get_list(path)
    return scalcsio.mec_load(path, meclist[0][0])


class TestRealFiles:
    """Need .mec files from outside the repository; skipped without them."""

    def test_a_file_that_stores_no_names(self):
        """DREMIS01.MEC: five states, every name field blank."""
        mec = _load('DREMIS01.MEC')
        assert sorted(mec.state_names) == ['S1', 'S2', 'S3', 'S4', 'S5']

    def test_positional_names_follow_the_numbers_on_the_stored_drawing(self):
        """DREMIS01.MEC is drawn as   C5--C4--C3
                                           |   |
                                          O1--O2   so S5 is top left."""
        layout = _load('DREMIS01.MEC').layout
        assert layout['S5'] == (0, 0)
        assert layout['S1'] == (2, 2)

    def test_names_padded_with_nuls(self):
        """qmechs.mec stores real names, padded with NUL bytes."""
        mec = _load('qmechs.mec')
        assert set(mec.state_names) == {'AR*', 'A2R*', 'AR', 'A2R', 'R'}
        assert mec.state_index('A2R*') == 1

    def test_some_names_stored_and_some_blank(self):
        """qmechsDC1.mec names its open states and leaves the shut ones
        blank."""
        mec = _load('qmechsDC1.mec')
        assert set(mec.state_names) == {'AR*', 'A2R*', 'S3', 'S4', 'S5'}
