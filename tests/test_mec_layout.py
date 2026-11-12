"""Tests for the mechanism drawing stored in a .mec file.

A .mec file carries, with each mechanism, the "chess board" on which the DOS
programs drew it: a grid of two-character cells holding state numbers and
line-drawing characters. The reader used to read it and throw it away. It is
now kept as ``Mechanism.layout``, the position of each state on that grid.

Only the positions are taken. The lines on the board are not parsed: the
transitions of a mechanism are its rates, and the boards in circulation draw
their connections with corner pieces, long runs and, for the cube mechanisms,
an oblique projection that cannot be read back reliably.

The boards below are copied from real files, so the tests need no .mec file.
"""
import os

import pytest

from scalcs import scalcsio
from scalcs.samples.samples import CH82


def board(*lines):
    """Text rows -> rows of two-byte cells, as they are stored in the file.

    The files are written in code page 437, where the horizontal and vertical
    line-drawing characters are 0xC4 and 0xB3.
    """
    width = max(len(line) for line in lines)
    width += width % 2
    rows = []
    for line in lines:
        raw = line.ljust(width).encode('cp437')
        rows.append([raw[i:i + 2] for i in range(0, width, 2)])
    return rows


# bgtv2a.mec: the flip mechanism of Burzomato et al. (2004)
FLIP_BOARD = board('10──C9──C8──C7',
                   '     │   │   │',
                   '    C6──C5──C4',
                   '     │   │   │',
                   '    O1  O2  O3')
FLIP_NAMES = ['AF*', 'A2F*', 'A3F*', 'A3F', 'A2F', 'AF',
              'A3R', 'A2R', 'AR', 'R']           # state 1 first


class TestLayoutFromBoard:

    def test_every_state_is_placed(self):
        layout = scalcsio._layout_from_board(FLIP_BOARD, FLIP_NAMES)
        assert set(layout) == set(FLIP_NAMES)

    def test_positions_are_column_then_row_with_the_top_row_zero(self):
        layout = scalcsio._layout_from_board(FLIP_BOARD, FLIP_NAMES)
        assert layout['R'] == (0, 0)             # state 10, top left
        assert layout['A3R'] == (6, 0)           # state 7, top right
        assert layout['AF'] == (2, 2)            # state 6, under AR
        assert layout['A3F*'] == (6, 4)          # state 3, bottom right

    def test_states_in_one_conformation_share_a_row(self):
        layout = scalcsio._layout_from_board(FLIP_BOARD, FLIP_NAMES)
        assert {layout[n][1] for n in ('R', 'AR', 'A2R', 'A3R')} == {0}
        assert {layout[n][1] for n in ('AF', 'A2F', 'A3F')} == {2}
        assert {layout[n][1] for n in ('AF*', 'A2F*', 'A3F*')} == {4}

    def test_a_two_digit_number_written_with_a_leading_zero(self):
        """Auerbach.mec writes state 3 as '03'."""
        layout = scalcsio._layout_from_board(
            board('03──O2──C1'), ['s1', 's2', 's3'])
        assert layout == {'s3': (0, 0), 's2': (2, 0), 's1': (4, 0)}

    def test_two_digit_numbers_written_bare(self):
        """SSNMDA.MEC: fifteen states, 10 to 15 with no letter."""
        names = ['s%d' % n for n in range(1, 16)]
        layout = scalcsio._layout_from_board(
            board('15──14──13',
                  ' │   │   │',
                  '12──11──10',
                  ' │   │   │',
                  'C9──C8──C7──C6──C3',
                  '         │   │',
                  '        C5──C4',
                  '         │   │',
                  '        O2  O1'), names)
        assert len(layout) == 15
        assert layout['s15'] == (0, 0)
        assert layout['s10'] == (4, 2)
        assert layout['s1'] == (6, 8)

    def test_state_ten_written_as_a_zero(self):
        """SCH5S15.MEC has ten states and writes the tenth as 'C0'."""
        names = ['s%d' % n for n in range(1, 11)]
        layout = scalcsio._layout_from_board(
            board('C0──C9──C8──C7──O1',
                  ' │   │   │   │   │',
                  'C6──C5──C4──C3──C2'), names)
        assert len(layout) == 10
        assert layout['s10'] == (0, 0)

    def test_a_drawing_that_does_not_place_every_state_gives_no_layout(self):
        """The cube mechanisms are drawn in an oblique projection that
        overwrites some of the state cells. Part of a layout is no use to
        anything that draws, so it is all of the states or None.
        """
        assert scalcsio._layout_from_board(
            board('O1──C2'), ['s1', 's2', 's3']) is None

    def test_corner_pieces_and_long_runs_are_not_states(self):
        """remis.MEC joins C7 to C5 with a line that turns a corner."""
        names = ['s%d' % n for n in range(1, 12)]
        layout = scalcsio._layout_from_board(
            board('11──C9──C6──O2',
                  ' │   │   │',
                  '10──C8──C5──O1──C4',
                  ' │       │',
                  'C7───────┘',
                  ' │',
                  'O3'), names)
        assert len(layout) == 11
        assert layout['s7'] == (0, 4)
        assert layout['s5'] == (4, 2)

    def test_positions_start_at_the_origin(self):
        """A board with blank columns to the left is shifted, so layouts from
        different files are comparable."""
        names = ['s1', 's2']
        layout = scalcsio._layout_from_board(
            board('    O1──C2'), names)
        assert layout == {'s1': (0, 0), 's2': (2, 0)}

    def test_a_blank_board_gives_no_layout(self):
        assert scalcsio._layout_from_board(board('      ', '      '),
                                           ['s1']) is None
        assert scalcsio._layout_from_board([], ['s1']) is None

    def test_states_that_share_a_name_give_no_layout(self):
        """Some older files store no state names, so every state has the
        same blank one. Positions keyed by name cannot tell them apart."""
        assert scalcsio._layout_from_board(
            board('C3──C2──O1'), ['', '', '']) is None

    def test_a_number_with_no_such_state_is_skipped(self):
        """A drawing is not worth failing to load a mechanism over."""
        layout = scalcsio._layout_from_board(board('O1──C2──C9'),
                                             ['s1', 's2'])
        assert layout == {'s1': (0, 0), 's2': (2, 0)}


class TestMechanismLayout:

    def test_a_mechanism_defined_in_python_has_no_layout(self):
        assert CH82().layout is None


MEC_DIR = os.environ.get('SCALCS_MEC_DIR', '')
BGTV2A = os.path.join(MEC_DIR, 'bgtv2a.mec')


@pytest.mark.skipif(not os.path.isfile(BGTV2A),
                    reason='set SCALCS_MEC_DIR to a folder holding bgtv2a.mec')
class TestLayoutFromARealFile:

    @pytest.fixture(scope='class')
    def mec(self):
        _, meclist, _ = scalcsio.mec_get_list(BGTV2A)
        return scalcsio.mec_load(BGTV2A, meclist[0][0])

    def test_every_state_of_the_mechanism_is_placed(self, mec):
        assert set(mec.layout) == {s.name for s in mec.States}

    def test_it_is_the_layout_of_the_published_figure(self, mec):
        """Three rows: resting, flipped, open. The file calls the flipped
        states AP, A2P, A3P and the open ones AR*, A2R*, A3R*."""
        rows = {}
        for name, (_, row) in mec.layout.items():
            rows.setdefault(row, set()).add(name)
        assert rows[0] == {'R', 'AR', 'A2R', 'A3R'}
        assert rows[2] == {'AP', 'A2P', 'A3P'}
        assert rows[4] == {'AR*', 'A2R*', 'A3R*'}
