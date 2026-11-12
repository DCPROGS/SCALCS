"""Draw kinetic mechanisms as reaction schemes.

The style is the one Colquhoun & Lape (2012) use for Figs. 3, 7 and 8: the
states set out on a grid, each reversible step drawn as a pair of harpoon
arrows lying close together, with a number written against each direction -
a rate constant, a transition probability or a transition frequency.

    from scalcs import schemes
    from scalcs.samples.samples import CH82

    schemes.draw(CH82(), values='probability', conc=100e-9, numbers=True)

Which states are connected is taken from the rates of the mechanism and from
nothing else. Where the states go - the layout - is the only thing the caller
supplies, and usually not even that:

  1. ``layout=`` given explicitly, as rows of state names (top row first,
     None for a gap) or as ``{name: (column, row)}``;
  2. otherwise ``mec.layout``, the drawing stored with a mechanism read from
     a .mec file;
  3. otherwise a ready-made layout from :data:`LAYOUTS`, matched on the names
     of the states, which covers every mechanism in ``scalcs.samples``.

Inferring the connections from the layout instead - joining neighbours on the
grid - is the obvious shortcut and the wrong one: it once drew the ten-state
flip mechanism with thirteen connections instead of eleven, joining open
states that do not interconvert.

Everything except :func:`draw` is plain arithmetic, so the geometry can be
computed, inspected and tested without matplotlib, which is an optional
dependency: ``pip install scalcs[plot]``.
"""
import math
import re
from dataclasses import dataclass

import numpy as np

from scalcs import qmatlib as qml


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Inches of figure per data unit at which the style's font sizes were tuned:
# Figs. 7 and 8 of Colquhoun & Lape (2012) set as two panels of a 7.6 x 9.4
# inch figure.
REFERENCE_SCALE = 0.734

@dataclass
class Style:
    """How a scheme is drawn.

    The defaults were measured from the published Figs. 3, 7 and 8 of
    Colquhoun & Lape (2012). Every length is a fraction of `spacing`, the
    distance between neighbouring states in data units, so changing
    `spacing` rescales the whole drawing without retuning anything else.
    """
    spacing: float = 1.4         # between neighbouring columns
    row_spacing: float = None    # between rows; None means the same as spacing.
                                 # The published Fig. 3 uses 1.5 times spacing.

    pair_offset: float = 0.026   # half the gap between the two arrows of a pair
    label_gap: float = 0.30      # clearance left round each state name
    head_length: float = 0.20
    head_width: float = 0.046
    label_pad: float = 0.065     # from an arrow to the number written by it
    line_width: float = 1.1

    # Font sizes and the line width are those of the published figures, at
    # REFERENCE_SCALE. Lengths are in data units but text is in points, so a
    # scheme drawn into a smaller panel would have its names overrun the
    # arrows; with scale_text they shrink with the axes instead. They are
    # never enlarged.
    state_fontsize: float = 19
    value_fontsize: float = 9
    number_fontsize: float = 9
    scale_text: bool = True

    open_colour: str = '#ed1c24'      # the paper's red
    shut_colour: str = 'black'
    arrow_colour: str = 'black'
    number_colour: str = '#1c9ad6'    # the paper's blue state numbering

    def __post_init__(self):
        if self.row_spacing is None:
            self.row_spacing = self.spacing


# ---------------------------------------------------------------------------
# Ready-made layouts
# ---------------------------------------------------------------------------

# Rows of state names, top row first, None for an empty cell. They follow the
# convention DC used in the drawings stored in his .mec files: binding from
# left to right along the resting row, a change of conformation downwards,
# open states at the bottom.
LAYOUTS = {
    'CO': [['C'],
           ['O']],
    'RFO': [['R', 'F'],
            [None, 'F*']],
    'RFFOO': [['R', 'F', 'FF'],
              [None, 'F*', 'FF*']],
    'CCO': [['R', 'AR'],
            [None, 'AR*']],
    'CCOB': [['R', 'AR', None],
             [None, 'AR*', 'AR*B']],
    'CCOD': [['R', 'AR', None],
             [None, 'AR*', 'AD']],
    'CH82': [['R', 'AR', 'A2R'],
             [None, 'AR*', 'A2R*']],
    'CHME97': [['R', 'AR', 'A2R', 'A2D'],
               [None, None, 'A2R*', None]],
    'GlyR_flip': [['R', 'AR', 'A2R', 'A3R'],
                  [None, 'AF', 'A2F', 'A3F'],
                  [None, 'AF*', 'A2F*', 'A3F*']],
    'AChR_diamond': [[None, 'AR*a', None, None],
                     [None, 'ARa', None, None],
                     ['R', None, 'A2R', 'A2R*'],
                     [None, 'ARb', None, None],
                     [None, 'AR*b', None, None]],
    'AChR_diamond_desens': [[None, 'AR*a', None, None, None],
                            [None, 'ARa', None, None, None],
                            ['R', None, 'A2R', 'A2R*', 'A2D'],
                            [None, 'ARb', None, None, None],
                            [None, 'AR*b', None, None, None]],
    # Every state joined to every other, so the diagonals cross. Tried as a
    # triangle with a state at its centre, which avoids the crossing but
    # leaves three steps in a 53-degree corner, where their numbers collide.
    # The square leaves 45 degrees between neighbouring steps.
    'fully_connected_cycle': [['C1', 'C2'],
                              ['O1', 'O2']],
    'six_cycles_mec': [['A', 'B', 'C', 'D'],
                       ['E', 'F', 'G', 'H'],
                       ['I', 'J', 'K', 'L']],
}


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def format_number(value):
    """A number written the way the published figures write them.

    Plain decimals with trailing zeros removed, 1.0 as a bare 1; thousands
    printed whole, which is what makes a frequency of 6471.73 per second come
    out as 6472. Values of 1e5 and above - association rate constants, in
    practice - get a power of ten, as do values too small for seven decimal
    places.
    """
    value = float(value)
    if value == 0:
        return '0'
    if abs(value) >= 1e5:
        return _power_of_ten(value)
    if abs(value) >= 1000:
        return '%.0f' % value
    text = '%.3g' % value
    if 'e' in text:                       # below 1e-4
        text = '%.7f' % value
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    if text in ('0', '-0'):               # below 5e-8
        return _power_of_ten(value)
    return text


def _power_of_ten(value):
    mantissa, exponent = ('%.3g' % value).split('e')
    return r'$%s\times10^{%d}$' % (mantissa, int(exponent))


def state_label(name):
    """A state name as mathtext: digits after a letter become subscripts and
    an asterisk a superscript, so 'A2F*' is written A2F* with a subscript 2
    and a raised star. Names with any other character are returned as they
    are, rather than risk mathtext misreading them.
    """
    if not re.fullmatch(r'[A-Za-z0-9*]+', name):
        return name
    body = re.sub(r'([A-Za-z])(\d+)', r'\1_{\2}', name)
    body = body.replace('*', '^{*}')
    return r'$\mathrm{%s}$' % body


def state_colours(mec, colours=None, style=None):
    """{name: colour}: open states in the style's open colour, the rest in
    its shut colour, then any given in `colours` - which is how the flipped
    conformation of Figs. 7 and 8 gets its green."""
    style = style or Style()
    out = {state.name: (style.open_colour if state.statetype == 'A'
                        else style.shut_colour)
           for state in mec.States}
    if colours:
        _check_names(mec, colours, 'colours')
        out.update(colours)
    return out


def state_numbers(mec, numbers):
    """{name: number} to write by each state.

    False writes none. True numbers the states in Q-matrix order, from 1,
    which for a mechanism read from a .mec file is the numbering of the file.
    A dict gives any other numbering - a paper's own, which need not be
    SCALCS's: the published flip mechanism numbers two of its states the
    other way round.
    """
    if not numbers:
        return {}
    if numbers is True:
        return {name: i + 1 for i, name in enumerate(mec.state_names)}
    _check_names(mec, numbers, 'numbers')
    return dict(numbers)


def _check_names(mec, given, what):
    unknown = sorted(set(given) - set(mec.state_names))
    if unknown:
        raise ValueError('%s names %s, which %s not a state of this mechanism '
                         '(its states are %s)'
                         % (what, ', '.join(unknown),
                            'is' if len(unknown) == 1 else 'are',
                            ', '.join(mec.state_names)))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class Arrow:
    """One direction of one step: source -> target.

    `barb` is the filled triangle at the head: the tip of the arrow, the
    point of the barb, and where the barb rejoins the shaft.
    """
    source: str
    target: str
    start: tuple
    end: tuple
    barb: tuple
    label: str = None
    label_xy: tuple = None
    label_ha: str = 'center'
    label_va: str = 'center'


@dataclass
class SchemeGeometry:
    positions: dict              # {name: (x, y)} in data units, y upwards
    arrows: list                 # one Arrow per rate of the mechanism


def geometry(mec, layout=None, values=None, conc=None, style=None):
    """Where every state, arrow and number goes.

    Parameters
    ----------
    mec : Mechanism
    layout : list of rows, or dict, optional
        Rows of state names, top row first, None for an empty cell; or
        ``{name: (column, row)}`` with row 0 at the top. If omitted, the
        layout stored with the mechanism is used, then a ready-made one from
        :data:`LAYOUTS`.
    values : None, 'rates', 'probability', 'frequency', or a function
        What to write against each arrow. 'rates' writes the rate constants;
        'probability' the transition probabilities pi_ij, Eq. 2 of
        Colquhoun & Lape (2012); 'frequency' the transition frequencies at
        equilibrium, f_ij = p_i q_ij, their Eq. 3. A function
        ``f(source, target)`` may return any text, or None to leave that
        direction unlabelled.
    conc : float, optional
        Agonist concentration (M) for 'probability' and 'frequency'. Needed
        if any rate depends on concentration. The mechanism is returned to
        the concentration it had.
    style : Style, optional

    Returns
    -------
    SchemeGeometry
    """
    style = style or Style()
    cells = _resolve_layout(mec, layout)
    positions = {name: (column * style.spacing, -row * style.row_spacing)
                 for name, (column, row) in cells.items()}
    _check_no_step_crosses_a_state(mec, positions, style)

    label = _labeller(mec, values, conc)
    arrows = [_arrow(source, target, positions, style,
                     label(source, target) if label else None)
              for source, target in mec.rates_by_edge()]
    return SchemeGeometry(positions, arrows)


def _arrow(source, target, positions, style, label):
    """Every arrow sits to its own left of the line between the two states
    and carries its barb on its own left. The two arrows of a pair run in
    opposite directions, so that puts each barb on the outside of the pair,
    at whatever angle the step runs, with no need to decide which direction
    is 'forward'.
    """
    (xa, ya), (xb, yb) = positions[source], positions[target]
    length = math.hypot(xb - xa, yb - ya)
    ux, uy = (xb - xa) / length, (yb - ya) / length      # along the arrow
    nx, ny = -uy, ux                                     # to its left

    gap = style.label_gap * style.spacing
    offset = style.pair_offset * style.spacing
    start = (xa + ux * gap + nx * offset, ya + uy * gap + ny * offset)
    end = (xb - ux * gap + nx * offset, yb - uy * gap + ny * offset)

    head = style.head_length * style.spacing
    width = style.head_width * style.spacing
    back = (end[0] - ux * head, end[1] - uy * head)
    barb = (end,
            (back[0] + nx * width, back[1] + ny * width),
            (end[0] - ux * head * 0.40, end[1] - uy * head * 0.40))

    arrow = Arrow(source, target, start, end, barb, label)
    if label is not None:
        _place_label(arrow, ux, uy, nx, ny, style)
    return arrow


# An arrow within about 25 degrees of horizontal or vertical has its label
# placed as Figs. 7 and 8 place them; anything steeper is a diagonal.
_NEARLY_ALIGNED = 0.9


def _place_label(arrow, ux, uy, nx, ny, style):
    """Beside the arrow, on the outside, near its tail - where Figs. 7 and 8
    put them.

    Horizontal: centred a fifth of the way along, above or below. Vertical:
    beside the arrow, starting level with the tail and reading towards the
    head. Diagonal: a fifth of the way along, anchored by the corner of the
    text nearest the arrow, so that all of the text lies outside the pair -
    centred, half of it would lie across the shaft.
    """
    (sx, sy), (ex, ey) = arrow.start, arrow.end
    pad = style.label_pad * style.spacing
    if abs(ux) >= _NEARLY_ALIGNED:
        along = 0.20
        arrow.label_ha = 'center'
        arrow.label_va = 'bottom' if ny > 0 else 'top'
    elif abs(uy) >= _NEARLY_ALIGNED:
        along = 0.04
        arrow.label_ha = 'left' if nx > 0 else 'right'
        arrow.label_va = 'top' if uy < 0 else 'bottom'
    else:
        along = 0.20
        arrow.label_ha = 'left' if nx > 0 else 'right'
        arrow.label_va = 'bottom' if ny > 0 else 'top'
    arrow.label_xy = (sx + along * (ex - sx) + nx * pad,
                      sy + along * (ey - sy) + ny * pad)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _resolve_layout(mec, layout):
    if layout is not None:
        cells = _cells(layout)
    elif getattr(mec, 'layout', None) is not None:
        cells = _compact(mec.layout)
    else:
        cells = _ready_made(mec)
        if cells is None:
            raise ValueError(
                'no layout for this mechanism. Pass one as layout=[[...], '
                '[...]], rows of state names with the top row first and None '
                'for an empty cell, or as layout={name: (column, row)}. Its '
                'states are %s.' % ', '.join(mec.state_names))
    _check_layout(mec, cells)
    return cells


def _cells(layout):
    """{name: (column, row)} from either form of layout."""
    if isinstance(layout, dict):
        return {name: tuple(cell) for name, cell in layout.items()}
    cells = {}
    for row, names in enumerate(layout):
        for column, name in enumerate(names):
            if name is None:
                continue
            if name in cells:
                raise ValueError('%s appears twice in the layout' % name)
            cells[name] = (column, row)
    return cells


def _compact(board_layout):
    """A layout read from a .mec drawing, on a grid of unit steps.

    The drawings put states on every other cell, with the connecting lines
    in between. The common step is divided out of each axis, so a drawing
    and the equivalent rows of names give the same positions.
    """
    columns = [column for column, _ in board_layout.values()]
    rows = [row for _, row in board_layout.values()]
    left, top = min(columns), min(rows)
    step_x = math.gcd(*[c - left for c in columns]) or 1
    step_y = math.gcd(*[r - top for r in rows]) or 1
    return {name: ((column - left) // step_x, (row - top) // step_y)
            for name, (column, row) in board_layout.items()}


def _ready_made(mec):
    """The entry of LAYOUTS with exactly this mechanism's state names."""
    names = set(mec.state_names)
    for rows in LAYOUTS.values():
        cells = _cells(rows)
        if set(cells) == names:
            return cells
    return None


def _check_layout(mec, cells):
    names = set(mec.state_names)
    missing = [n for n in mec.state_names if n not in cells]
    if missing:
        raise ValueError('the layout gives no position for %s'
                         % ', '.join(missing))
    _check_names(mec, cells, 'the layout')
    taken = {}
    for name, cell in cells.items():
        if cell in taken:
            raise ValueError('%s and %s are in the same place, %s'
                             % (taken[cell], name, cell))
        taken[cell] = name


def _check_no_step_crosses_a_state(mec, positions, style):
    """A step whose arrows pass over a third state would be read as two
    steps through it - a transition the mechanism does not have, drawn."""
    clearance = style.label_gap * style.spacing
    done = set()
    for source, target in mec.rates_by_edge():
        pair = frozenset((source, target))
        if pair in done:
            continue
        done.add(pair)
        a, b = np.array(positions[source]), np.array(positions[target])
        for name, point in positions.items():
            if name in pair:
                continue
            t = np.dot(np.array(point) - a, b - a) / np.dot(b - a, b - a)
            if 0 < t < 1:
                nearest = a + t * (b - a)
                if np.hypot(*(np.array(point) - nearest)) < clearance:
                    raise ValueError(
                        'the step between %s and %s would be drawn through '
                        '%s; move one of them in the layout'
                        % (source, target, name))


# ---------------------------------------------------------------------------
# Values written against the arrows
# ---------------------------------------------------------------------------

VALUES = ('rates', 'probability', 'frequency')


def _labeller(mec, values, conc):
    if values is None:
        return None
    if callable(values):
        return values
    if values not in VALUES:
        raise ValueError("values must be one of %s, or a function "
                         "f(source, target) returning text; got %r"
                         % (', '.join(repr(v) for v in VALUES), values))
    if values == 'rates':
        rates = mec.rates_by_edge()
        return lambda a, b: format_number(rates[(a, b)].unit_rate())

    Q = _q_at(mec, conc)
    if values == 'probability':
        matrix = qml.transition_probability(Q)
    else:
        matrix = qml.transition_frequency(Q)
    return lambda a, b: format_number(
        matrix[mec.state_index(a), mec.state_index(b)])


def _q_at(mec, conc):
    """The Q matrix at `conc`, leaving the mechanism as it was."""
    if conc is None:
        if mec._effdict:
            raise ValueError(
                'the rates of this mechanism depend on concentration, so '
                'transition probabilities and frequencies need one: pass '
                'conc= in M')
        return np.array(mec.Q, dtype=float, copy=True)
    saved = dict(mec._effdict)
    try:
        mec.set_eff('c', conc)
        return np.array(mec.Q, dtype=float, copy=True)
    finally:
        mec.set_effdict(saved)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw(mec, ax=None, layout=None, values=None, conc=None, colours=None,
         numbers=False, style=None):
    """Draw the mechanism; return the axes.

    Parameters are those of :func:`geometry`, plus:

    ax : matplotlib Axes, optional
        Drawn into a new figure, sized to the scheme, if not given. Text and
        lines are scaled to the size of the axes at the time of drawing; see
        :class:`Style`.
    colours : dict, optional
        {name: colour} for any states not to be drawn in the default open or
        shut colour.
    numbers : bool or dict
        State numbers to write in blue beside each state; see
        :func:`state_numbers`.

    Requires matplotlib (``pip install scalcs[plot]``).
    """
    import matplotlib.pyplot as plt

    style = style or Style()
    scheme = geometry(mec, layout=layout, values=values, conc=conc,
                      style=style)
    colour = state_colours(mec, colours, style)
    number = state_numbers(mec, numbers)

    xs = [x for x, _ in scheme.positions.values()]
    ys = [y for _, y in scheme.positions.values()]
    margin = 0.6 * style.spacing
    width = max(xs) - min(xs) + 2 * margin
    height = max(ys) - min(ys) + 2 * margin
    if ax is None:
        # axes filling a figure sized so the scheme is at the reference scale
        figure = plt.figure(figsize=(width * REFERENCE_SCALE,
                                     height * REFERENCE_SCALE))
        ax = figure.add_axes([0, 0, 1, 1])

    # Limits and aspect first, so the size of the axes - and from it the
    # scale of the text - is known before anything is drawn. Equal aspect,
    # or the harpoons shear and a pair stops looking parallel.
    ax.set_aspect('equal')
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.axis('off')
    scale = _text_scale(ax, width) if style.scale_text else 1.0

    for arrow in scheme.arrows:
        (sx, sy), (ex, ey) = arrow.start, arrow.end
        ax.plot([sx, ex], [sy, ey], color=style.arrow_colour,
                lw=style.line_width * scale, solid_capstyle='butt', zorder=2)
        ax.add_patch(plt.Polygon(arrow.barb, closed=True,
                                 facecolor=style.arrow_colour,
                                 edgecolor='none', zorder=3))
        if arrow.label is not None:
            ax.text(*arrow.label_xy, arrow.label, ha=arrow.label_ha,
                    va=arrow.label_va, fontsize=style.value_fontsize * scale)

    for name, (x, y) in scheme.positions.items():
        ax.text(x, y, state_label(name), ha='center', va='center',
                fontsize=style.state_fontsize * scale, color=colour[name])
        if name in number:
            ax.text(x + 0.093 * style.spacing, y - 0.079 * style.spacing,
                    str(number[name]), ha='left', va='top',
                    fontsize=style.number_fontsize * scale,
                    color=style.number_colour)
    return ax


def _text_scale(ax, width):
    """Inches per data unit of `ax`, relative to REFERENCE_SCALE, and never
    more than 1.

    Text shrinks with a panel smaller than the reference, so names do not
    overrun the arrows; it is not enlarged in a bigger one, where it would
    only make the numbers of neighbouring steps crowd each other.

    apply_aspect() resolves the box that equal aspect leaves the axes, so the
    horizontal scale is also the vertical one.
    """
    ax.apply_aspect()
    inches = ax.get_position().width * ax.figure.get_figwidth()
    return min(1.0, inches / width / REFERENCE_SCALE)
