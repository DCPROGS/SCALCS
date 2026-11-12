"""Tests for scalcs.schemes, the kinetic-scheme drawer.

Most of these test the geometry - where the states go and where every arrow
starts, ends and points - which is plain arithmetic and needs no matplotlib.
Only TestDrawing renders anything.

The drawing style is the one Colquhoun & Lape (2012) use for Figs. 3, 7 and
8: closely paired harpoon arrows, the single barb of each on the outside of
its pair.
"""
import inspect
import math
import subprocess
import sys

import numpy as np
import pytest

from scalcs import qmatlib as qml
from scalcs import schemes
from scalcs.mechanism import Mechanism, Rate, State
from scalcs.samples import samples
from scalcs.samples.samples import CCO, CH82, GlyR_flip


def _sample_mechanisms():
    """Every mechanism in scalcs.samples that builds without arguments, so a
    sample added later without a layout fails a test here."""
    out = []
    for name, factory in inspect.getmembers(samples, inspect.isfunction):
        try:
            mec = factory()
        except TypeError:
            continue
        if isinstance(mec, Mechanism):
            out.append((name, mec))
    return out


SAMPLES = _sample_mechanisms()


def _connections(mec):
    return {frozenset(edge) for edge in mec.rates_by_edge()}


def _pairs(geometry):
    by_edge = {}
    for arrow in geometry.arrows:
        by_edge.setdefault(frozenset((arrow.source, arrow.target)), []).append(arrow)
    return [arrows for arrows in by_edge.values() if len(arrows) == 2]


def _unit(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    return dx / length, dy / length


def _side(point, start, end):
    """Signed distance of `point` to the left of the line start -> end."""
    ux, uy = _unit(start, end)
    return (point[0] - start[0]) * -uy + (point[1] - start[1]) * ux


def _two_state(names=('X', 'Y')):
    a, b = State('B', names[0], 0.0), State('A', names[1], 50e-12)
    return Mechanism([Rate(10.0, a, b, name='on'), Rate(100.0, b, a, name='off')])


# ---------------------------------------------------------------------------

class TestTopologyComesFromTheMechanism:
    """The connections drawn are the rates of the mechanism, never inferred
    from where the states have been put."""

    def test_one_arrow_per_rate(self):
        mec = GlyR_flip()
        assert len(schemes.geometry(mec).arrows) == len(mec.Rates)

    def test_the_flip_mechanism_has_eleven_connections_not_thirteen(self):
        """The open states A F*, A2F*, A3F* sit side by side but do not
        interconvert. A drawer that joined neighbours on a grid drew two
        binding steps between them that the mechanism does not have."""
        drawn = {frozenset((a.source, a.target))
                 for a in schemes.geometry(GlyR_flip()).arrows}
        assert len(drawn) == 11
        assert frozenset(('AF*', 'A2F*')) not in drawn
        assert frozenset(('A2F*', 'A3F*')) not in drawn

    @pytest.mark.parametrize('name, mec', SAMPLES, ids=[n for n, _ in SAMPLES])
    def test_arrows_join_exactly_the_states_the_rates_join(self, name, mec):
        drawn = {(a.source, a.target) for a in schemes.geometry(mec).arrows}
        assert drawn == set(mec.rates_by_edge())


class TestPairs:
    """The two arrows of a reversible step, at whatever angle it runs."""

    @pytest.fixture(scope='class', params=['CH82', 'GlyR_flip',
                                           'AChR_diamond',
                                           'fully_connected_cycle'])
    def geometry(self, request):
        return schemes.geometry(getattr(samples, request.param)())

    def test_the_two_arrows_run_in_opposite_directions(self, geometry):
        for first, second in _pairs(geometry):
            ux, uy = _unit(first.start, first.end)
            vx, vy = _unit(second.start, second.end)
            assert ux * vx + uy * vy == pytest.approx(-1.0)

    def test_they_are_separated_by_twice_the_pair_offset(self, geometry):
        style = schemes.Style()
        expected = 2 * style.pair_offset * style.spacing
        for first, second in _pairs(geometry):
            assert abs(_side(second.start, first.start, first.end)) == \
                pytest.approx(expected)

    def test_each_barb_is_on_the_outside_of_its_pair(self, geometry):
        """The barb of each arrow points away from the other arrow."""
        for first, second in _pairs(geometry):
            for arrow, other in ((first, second), (second, first)):
                tip_of_barb = arrow.barb[1]
                other_side = _side(other.start, arrow.start, arrow.end)
                barb_side = _side(tip_of_barb, arrow.start, arrow.end)
                assert barb_side * other_side < 0

    def test_arrows_stop_short_of_the_state_labels(self, geometry):
        style = schemes.Style()
        clearance = style.label_gap * style.spacing
        for arrow in geometry.arrows:
            for point, name in ((arrow.start, arrow.source),
                                (arrow.end, arrow.target)):
                centre = geometry.positions[name]
                distance = math.hypot(point[0] - centre[0], point[1] - centre[1])
                assert distance >= clearance * 0.99


    def test_labels_on_a_diagonal_are_anchored_away_from_the_arrow(self):
        """Centred above a 45-degree shaft, half of the text would lie on the
        arrow. A diagonal label is anchored by the corner nearest its arrow,
        so all of it lies outside the pair."""
        geometry = schemes.geometry(samples.AChR_diamond(),
                                    values=lambda a, b: 'x')
        diagonals = 0
        for arrow in geometry.arrows:
            ux, uy = _unit(arrow.start, arrow.end)
            if min(abs(ux), abs(uy)) < 0.5:
                continue
            diagonals += 1
            nx, ny = -uy, ux
            assert arrow.label_ha == ('left' if nx > 0 else 'right')
            assert arrow.label_va == ('bottom' if ny > 0 else 'top')
        assert diagonals == 8

    def test_labels_on_grid_steps_are_placed_as_in_the_paper(self):
        """Horizontal: centred, above or below. Vertical: beside the arrow,
        reading from its tail towards its head."""
        geometry = schemes.geometry(CH82(), values=lambda a, b: 'x')
        by = {(a.source, a.target): a for a in geometry.arrows}
        assert (by[('R', 'AR')].label_ha, by[('R', 'AR')].label_va) ==             ('center', 'bottom')
        assert (by[('AR', 'R')].label_ha, by[('AR', 'R')].label_va) ==             ('center', 'top')
        assert (by[('AR', 'AR*')].label_ha, by[('AR', 'AR*')].label_va) ==             ('left', 'top')
        assert (by[('AR*', 'AR')].label_ha, by[('AR*', 'AR')].label_va) ==             ('right', 'bottom')


class TestLayout:

    def test_rows_of_names_are_given_top_row_first(self):
        geometry = schemes.geometry(CCO(), layout=[['R', 'AR'], [None, 'AR*']])
        assert geometry.positions['AR'][0] == geometry.positions['AR*'][0]
        assert geometry.positions['AR'][1] > geometry.positions['AR*'][1]

    def test_a_dict_of_column_and_row_works_the_same_way(self):
        rows = schemes.geometry(CCO(), layout=[['R', 'AR'], [None, 'AR*']])
        cells = schemes.geometry(CCO(), layout={'R': (0, 0), 'AR': (1, 0),
                                                'AR*': (1, 1)})
        assert rows.positions == cells.positions

    def test_the_drawing_stored_with_a_mec_file_is_used(self):
        """A .mec board puts states on every other cell. That spacing is
        taken out, so a board layout and the equivalent rows agree."""
        mec = CCO()
        mec.layout = {'R': (0, 0), 'AR': (2, 0), 'AR*': (2, 2)}
        from_file = schemes.geometry(mec)
        mec.layout = None
        given = schemes.geometry(mec, layout=[['R', 'AR'], [None, 'AR*']])
        assert from_file.positions == given.positions

    def test_a_layout_given_explicitly_wins_over_the_stored_one(self):
        mec = CCO()
        mec.layout = {'R': (0, 0), 'AR': (2, 0), 'AR*': (2, 2)}
        geometry = schemes.geometry(mec, layout=[['AR*', 'AR', 'R']])
        assert geometry.positions['AR*'][0] < geometry.positions['R'][0]

    @pytest.mark.parametrize('name, mec', SAMPLES, ids=[n for n, _ in SAMPLES])
    def test_every_sample_mechanism_has_a_ready_made_layout(self, name, mec):
        assert mec.layout is None
        geometry = schemes.geometry(mec)
        assert set(geometry.positions) == set(mec.state_names)

    def test_a_state_left_off_the_layout_is_named(self):
        with pytest.raises(ValueError, match='AR\\*'):
            schemes.geometry(CCO(), layout=[['R', 'AR']])

    def test_a_name_that_is_not_a_state_is_named(self):
        with pytest.raises(ValueError, match='A2R'):
            schemes.geometry(CCO(), layout=[['R', 'AR', 'A2R'], [None, 'AR*']])

    def test_two_states_in_one_place_is_an_error(self):
        with pytest.raises(ValueError, match='same place'):
            schemes.geometry(CCO(), layout={'R': (0, 0), 'AR': (1, 0),
                                            'AR*': (1, 0)})

    def test_an_arrow_drawn_through_another_state_is_an_error(self):
        """R -> AR would pass straight over AR*, and read as two steps that
        the mechanism does not have."""
        with pytest.raises(ValueError, match='AR\\*'):
            schemes.geometry(CCO(), layout=[['R', 'AR*', 'AR']])

    def test_with_no_layout_anywhere_the_error_says_how_to_give_one(self):
        with pytest.raises(ValueError, match='layout='):
            schemes.geometry(_two_state())


class TestValues:

    def test_nothing_is_written_by_default(self):
        assert all(a.label is None for a in schemes.geometry(CH82()).arrows)

    def test_transition_probabilities(self):
        mec = CH82()
        geometry = schemes.geometry(mec, values='probability', conc=100e-9)
        mec.set_eff('c', 100e-9)
        pi = qml.transition_probability(mec.Q)
        for arrow in geometry.arrows:
            i, j = mec.state_index(arrow.source), mec.state_index(arrow.target)
            assert arrow.label == schemes.format_number(pi[i, j])

    def test_transition_frequencies_at_equilibrium(self):
        mec = CH82()
        geometry = schemes.geometry(mec, values='frequency', conc=100e-9)
        mec.set_eff('c', 100e-9)
        f = qml.transition_frequency(mec.Q)
        for arrow in geometry.arrows:
            i, j = mec.state_index(arrow.source), mec.state_index(arrow.target)
            assert arrow.label == schemes.format_number(f[i, j])

    def test_rate_constants_need_no_concentration(self):
        mec = CH82()
        geometry = schemes.geometry(mec, values='rates')
        for arrow in geometry.arrows:
            expected = mec.rate(arrow.source, arrow.target).unit_rate()
            assert arrow.label == schemes.format_number(expected)

    def test_probabilities_need_a_concentration(self):
        """Without one the Q matrix is at whatever concentration was last
        set - 1 M if none was - and the figure would be silently wrong."""
        with pytest.raises(ValueError, match='conc'):
            schemes.geometry(CH82(), values='probability')

    def test_a_mechanism_without_ligand_needs_none(self):
        mec = samples.RFO()
        geometry = schemes.geometry(mec, values='probability')
        assert all(a.label is not None for a in geometry.arrows)

    def test_the_mechanisms_own_concentration_is_left_alone(self):
        mec = CH82()
        mec.set_eff('c', 1e-6)
        before = np.array(mec.Q, copy=True)
        schemes.geometry(mec, values='frequency', conc=100e-9)
        assert np.array(mec.Q) == pytest.approx(before)

    def test_a_function_may_label_some_directions_and_not_others(self):
        geometry = schemes.geometry(
            CCO(), values=lambda a, b: 'on' if (a, b) == ('R', 'AR') else None)
        labels = {(a.source, a.target): a.label for a in geometry.arrows}
        assert labels[('R', 'AR')] == 'on'
        assert labels[('AR', 'R')] is None

    def test_an_unknown_kind_of_value_is_an_error(self):
        with pytest.raises(ValueError, match='probability'):
            schemes.geometry(CH82(), values='probabilities')


class TestFormatNumber:
    """Numbers as the published figures print them."""

    @pytest.mark.parametrize('value, text', [
        (0.0, '0'),
        (1.0, '1'),
        (0.418, '0.418'),
        (0.0573, '0.0573'),
        (6471.7296, '6472'),         # f43: thousands print whole
        (1850.6, '1851'),
    ])
    def test_as_in_the_paper(self, value, text):
        assert schemes.format_number(value) == text

    def test_large_rate_constants_use_a_power_of_ten(self):
        assert schemes.format_number(1.66282e8) == r'$1.66\times10^{8}$'


class TestStateLabels:

    @pytest.mark.parametrize('name, label', [
        ('R', r'$\mathrm{R}$'),
        ('A2R', r'$\mathrm{A_{2}R}$'),
        ('A2F*', r'$\mathrm{A_{2}F^{*}}$'),
        ('AR*a', r'$\mathrm{AR^{*}a}$'),
    ])
    def test_digits_are_subscripts_and_stars_superscripts(self, name, label):
        assert schemes.state_label(name) == label

    def test_a_name_mathtext_would_mangle_is_left_as_it_is(self):
        assert schemes.state_label('A_2 R') == 'A_2 R'


class TestColoursAndNumbers:

    def test_open_states_are_red_and_shut_ones_black(self):
        colours = schemes.state_colours(CH82())
        style = schemes.Style()
        assert colours['AR*'] == colours['A2R*'] == style.open_colour
        assert colours['R'] == colours['AR'] == style.shut_colour

    def test_any_state_can_be_given_its_own_colour(self):
        colours = schemes.state_colours(GlyR_flip(),
                                        {'AF': 'green', 'A2F': 'green'})
        assert colours['AF'] == colours['A2F'] == 'green'
        assert colours['A3F'] == schemes.Style().shut_colour

    def test_numbers_true_numbers_states_in_q_matrix_order(self):
        numbers = schemes.state_numbers(CH82(), True)
        assert numbers == {'AR*': 1, 'A2R*': 2, 'AR': 3, 'A2R': 4, 'R': 5}

    def test_a_paper_s_own_numbering_can_be_given(self):
        numbers = schemes.state_numbers(GlyR_flip(), {'A3F': 4, 'AF': 6})
        assert numbers == {'A3F': 4, 'AF': 6}

    def test_no_numbers_by_default(self):
        assert schemes.state_numbers(CH82(), False) == {}


class TestImport:

    def test_importing_the_module_does_not_import_matplotlib(self):
        """matplotlib is an optional extra: everything but draw() has to work
        without it."""
        code = ('import sys; import scalcs.schemes; '
                'print("matplotlib" in sys.modules)')
        out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                             text=True, check=True)
        assert out.stdout.strip() == 'False'


class TestDrawing:

    @pytest.fixture(autouse=True)
    def _agg(self):
        pytest.importorskip('matplotlib')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        yield
        plt.close('all')

    @pytest.mark.parametrize('name, mec', SAMPLES, ids=[n for n, _ in SAMPLES])
    def test_every_sample_mechanism_draws(self, name, mec):
        import matplotlib.pyplot as plt
        ax = schemes.draw(mec)
        barbs = [p for p in ax.patches if isinstance(p, plt.Polygon)]
        assert len(barbs) == len(mec.Rates)
        texts = {t.get_text() for t in ax.texts}
        assert {schemes.state_label(n) for n in mec.state_names} <= texts

    def test_values_numbers_and_colours_reach_the_figure(self):
        mec = CH82()
        ax = schemes.draw(mec, values='rates', numbers=True,
                          colours={'R': 'blue'})
        texts = {t.get_text(): t for t in ax.texts}
        assert schemes.format_number(mec.rate('R', 'AR').unit_rate()) in texts
        assert '5' in texts
        assert texts[schemes.state_label('R')].get_color() == 'blue'

    def _state_fontsize(self, ax, name='R'):
        return next(t.get_fontsize() for t in ax.texts
                    if t.get_text() == schemes.state_label(name))

    def test_a_new_figure_is_drawn_at_the_reference_scale(self):
        """The style's font sizes are the ones measured from the published
        figures, and a figure draw() makes for itself reproduces them."""
        ax = schemes.draw(CH82())
        assert self._state_fontsize(ax) == pytest.approx(
            schemes.Style().state_fontsize, rel=0.02)

    def test_text_shrinks_with_the_axes_it_is_drawn_into(self):
        """Lengths are in data units and font sizes in points, so a scheme
        squeezed into a small panel would otherwise have its names overrun
        the arrows. Text is scaled with the axes instead."""
        import matplotlib.pyplot as plt
        # both panels smaller than the reference, so neither is capped
        big = schemes.draw(CH82(), ax=plt.figure(figsize=(3, 2.25)).add_subplot())
        small = schemes.draw(CH82(), ax=plt.figure(figsize=(1.5, 1.125)).add_subplot())
        assert self._state_fontsize(big) < schemes.Style().state_fontsize
        assert self._state_fontsize(small) / self._state_fontsize(big) ==             pytest.approx(0.5, rel=0.02)

    def test_text_is_never_enlarged_beyond_the_published_size(self):
        """A small scheme in a big panel keeps the published sizes; scaled up,
        the numbers of neighbouring steps crowd each other."""
        import matplotlib.pyplot as plt
        ax = schemes.draw(CCO(), ax=plt.figure(figsize=(10, 8)).add_subplot())
        assert self._state_fontsize(ax, 'AR') == schemes.Style().state_fontsize

    def test_scaling_can_be_turned_off(self):
        import matplotlib.pyplot as plt
        ax = schemes.draw(CH82(), ax=plt.figure(figsize=(4, 3)).add_subplot(),
                          style=schemes.Style(scale_text=False))
        assert self._state_fontsize(ax) == schemes.Style().state_fontsize

    def test_it_draws_into_the_axes_it_is_given(self):
        import matplotlib.pyplot as plt
        fig, (left, right) = plt.subplots(1, 2)
        assert schemes.draw(CCO(), ax=right) is right
        assert not left.texts
