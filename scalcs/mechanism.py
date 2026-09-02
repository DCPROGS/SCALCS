#!/usr/bin/python
"""
Class Mechanism represents a kinetic reaction scheme.

CH82: Colquhoun D, Hawkes AG (1982)
On the stochastic properties of bursts of single ion channel openings
and of clusters of bursts. Phil Trans R Soc Lond B 300, 1-59.

"""

from __future__ import annotations

# TODO: impose detailed microscopic reversibility.
# TODO: impose constrains (e.g. independent binding sites).
# TODO: fix certain rate constants while fitting.
# TODO: Check state numbers for consistency
# TODO: Update docstrings

import math
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.optimize as so

__all__ = [
    "identity",
    "multiply",
    "constrain_rate_multiple",
    "State",
    "Cycle",
    "Rate",
    "Graph",
    "Mechanism",
]

#import qmatlib as qml

def identity(rate: np.ndarray, effdict: dict[str, float]) -> float:
    """
    Return rate[0]. Used as default rate function if
    the rate doesn't depend on an effector.

    Parameters
    ----------
    rate : np.ndarray
        Array of rate constants; only rate[0] is used.
    effdict : dict[str, float]
        Effector name → value mapping (e.g. ``{'c': 200}``).
        Ignored by this function.

    Returns
    -------
    float
        rate[0]
    """
    return rate[0]

def multiply(rate: np.ndarray, effdict: dict[str, float]) -> float:
    """
    Multiply rate[0] by the single effector value.  Used as default rate
    function when the rate depends on exactly one effector.

    Parameters
    ----------
    rate : np.ndarray
        Array of rate constants; only rate[0] is used.
    effdict : dict[str, float]
        Effector name → value mapping (e.g. ``{'c': 200}``).

    Returns
    -------
    float
        rate[0] * effector_value
    """
    return rate[0] * list(effdict.values())[0]

def constrain_rate_multiple(rate: np.ndarray, factor: float) -> np.ndarray:
    """
    Constrain a rate constant to be a fixed multiple of another rate constant.

    Parameters
    ----------
    rate : np.ndarray
        The reference rate constant array.
    factor : float
        Multiplicative factor.

    Returns
    -------
    np.ndarray
        rate * factor
    """
    return rate * factor


class State:
    """
    Describes a single kinetic state of the channel.

    Attributes
    ----------
    statetype : str
        One of ``'A'`` (open), ``'B'`` (short-lived shut, within burst),
        ``'C'`` (long-lived shut, between bursts), ``'D'`` (desensitised).
    name : str
        Human-readable label, e.g. ``'A2R*'``.
    conductance : float
        Single-channel conductance in Siemens (0 for shut states).
    no : int or None
        Zero-based index within the Q matrix; assigned by
        :class:`Mechanism` during construction.
    """

    no: int | None  # set by Mechanism.sort_states()

    def __init__(
        self,
        statetype: str = '',
        name: str = '',
        conductance: float = 0.0,
    ) -> None:
        if statetype not in ('A', 'B', 'C', 'D'):
            raise RuntimeError("State has to be one of 'A', 'B', 'C' or 'D'")
        self.statetype: str = statetype
        self.name: str = name
        self.conductance: float = conductance
        self.no: int | None = None  # assigned in Mechanism.sort_states(); ZERO-based


class Cycle:
    """
    Describes a thermodynamic cycle within a kinetic mechanism.

    Attributes
    ----------
    states : list[str]
        Ordered list of state names forming the cycle,
        e.g. ``['A2R*', 'AR*', 'AR', 'A2R']``.
    mrconstr : list[str]
        Pair of state names ``[from, to]`` identifying the rate that is
        determined by microscopic reversibility.  Empty list when no
        constraint is active.
    """

    def __init__(
        self,
        states: list[str],
        mrconstr: list[str] | None = None,
    ) -> None:
        self.states: list[str] = states
        # Fix: avoid shared mutable default; None sentinel → empty list
        self.mrconstr: list[str] = mrconstr if mrconstr is not None else []


class Rate:
    """
    Describes a transition rate between two :class:`State` objects.

    Parameters
    ----------
    rateconstants : float or array-like
        Rate constant(s) passed to *func*.  A scalar is wrapped in a 1-D
        array automatically.
    State1 : State
        Source state of the transition.
    State2 : State
        Destination state of the transition.
    name : str
        Human-readable label, e.g. ``'beta2'``.
    eff : str or list[str] or None
        Effector name(s) that modulate this rate (e.g. ``'c'`` for
        concentration, ``'v'`` for voltage).  ``None`` for
        effector-independent rates.
    fixed : bool
        If ``True``, this rate is held constant during fitting.
    mr : bool
        If ``True``, this rate is determined by microscopic reversibility.
    func : Callable or None
        Rate equation ``f(rateconstants, effdict) -> float``.  If
        ``None``, :func:`identity` or :func:`multiply` is chosen
        automatically.
    limits : list
        ``[[lower, upper], ...]`` bounds for each rate constant.
    is_constrained : bool
        If ``True``, this rate is computed from another via
        *constrain_func*.
    constrain_func : Callable or None
        Function ``f(ref_rate, args) -> new_rate`` used when
        *is_constrained* is ``True``.
    constrain_args : list or None
        Arguments forwarded to *constrain_func*.
    """

    def __init__(
        self,
        rateconstants: float | list | np.ndarray,
        State1: State,
        State2: State,
        name: str = ' ',
        eff: str | list[str] | None = None,
        fixed: bool = False,
        mr: bool = False,
        func: Callable | None = None,
        limits: list | None = None,
        is_constrained: bool = False,
        constrain_func: Callable | None = None,
        constrain_args: list | None = None,
    ) -> None:
        # Fix: avoid shared mutable default for limits
        if limits is None:
            limits = []

        self._set_name(name)
        if not isinstance(State1, State) or not isinstance(State2, State):
            raise TypeError("DCPYPS: States have to be of class State")
        self.State1 = State1
        self.State2 = State2

        self._set_rateconstants(rateconstants)
        self._set_effectors(eff)
        self._set_limits(limits, impose=False)
        # List of effectors. Eg: 'c', 'v', ['Glu', 'Gly']
        
        self._set_func(func)

        self.fixed = fixed # for future expansion (fixed while fitting)
        self.mr = mr # for future expansion (set by microscopic reversibility)
        self.is_constrained = is_constrained
        self.constrain_func = constrain_func
        self.constrain_args = constrain_args
        # set by Mechanism.set_EC50_constraint(): this rate is computed from a
        # specified equilibrium EC50 together with all the other rates
        self.ec50_constrained = False

    @property
    def is_free(self):
        """True if this rate is a free parameter of a fit.

        A rate is free unless it is held fixed, is a multiple of another rate,
        is set by microscopic reversibility, or is computed from a specified
        EC50.
        """
        return not (self.fixed or self.is_constrained or self.mr
                    or self.ec50_constrained)

    def _set_name(self, name):
        self._name = name
    def _get_name(self):
        return self._name
    name = property(_get_name, _set_name)

    def _set_func(self, func):
        if func is None:
            # No function provided, set up a default function

            # Default functions only work for single rate constant
            if len(self._rateconstants) != 1:
                errmsg = "DCPYPS: More than one rate constant provided. " % self.name
                errmsg += "Can't use default rate function; please provide one.\n"
                raise RuntimeError(errmsg)

            if self._effectors[0] is not None:
                # single effector, use simple multiplication
                if len(self._effectors) == 1:
                    self._func = multiply
                else:
                    errmsg = "DCPYPS: Rate %s depends on more than one effector. " % self.name
                    errmsg += "Can't use default rate function; please provide one.\n"
                    raise RuntimeError(errmsg)
            # effector-independent, return rate[0]
            else:
                self._func = identity
        else:
            # TODO: sanity check of func
            self._func = func # f(ratepars, amount of effector); "Rate equation" if you wish
    def _get_func(self):
        return self._func
    func = property(_get_func, _set_func)

    def calc(self, effdict):
        return self._func(self._rateconstants, effdict)

    def unit_rate(self):
        # Set up a dictionary with all effectors set to 1:
        unitdict = {}
        for eff in self._effectors:
            unitdict[eff] = 1.0
        return self._func(self._rateconstants, unitdict)

    def _set_effectors(self, effectors):
        try:
            # test whether effector is a sequence:
            it = iter(effectors)
            self._effectors = effectors
        except TypeError:
            # if not, convert to single-itemed list:
            self._effectors = [effectors,]
    def _get_effectors(self):
        return self._effectors
    effectors = property(_get_effectors, _set_effectors)

    def _set_rateconstants(self, rateconstants):
        try:
            # test whether rateconstants is a sequence:
            it = iter(rateconstants)
            # test whether this is a numpy array:
            if isinstance(rateconstants, np.ndarray):
                self._rateconstants = rateconstants
            else:
                # else, convert:
                self._rateconstants = np.array(rateconstants)
        except TypeError:
            # if not, convert to single-itemed list:
            self._rateconstants = np.array([rateconstants,])
    def _get_rateconstants(self):
        return self._rateconstants
    rateconstants = property(_get_rateconstants, _set_rateconstants)

    def _set_limits(self, limits, impose=False):
        self._limits = limits
        self._check_limits()
        if impose:
            self._impose_limits()
    def _get_limits(self):
        return self._limits
    limits = property(_get_limits, _set_limits)
    def _check_limits(self):
        # sanity check for rate constant limits:
        if len(self._limits):
            # There must be as many limits as rate constants, except if there's only
            # one rate constant:
            if len(self._rateconstants)==1:
                err = "DCPYPS: If there's only one rate constant, limits\n"
                err += "can either be a list with upper and lower bounds\n"
                err += "(i.e. [lower, upper]) or a list of lists\n"
                err += "(i.e. [[lower, upper]]).\n"
                if len(self._limits)==2:
                    self._limits = [self._limits,]
                if len(self._limits) > 2:
                    raise RuntimeError(err)
                if len(self._limits)==1 and len(self._limits[0]) != 2:
                    raise RuntimeError(err)
            elif len(self._limits) != len(self._rateconstants):
                err = "DCPYPS: limits has to contain as many limit pairs as there are rate constants.\n"
                raise RuntimeError(err)
        else: # if no limits given- set default
            self._set_default_limits()
    def _impose_limits(self):
        if self._limits != []:
            for nr in range(len(self._rateconstants)):
                if self._rateconstants[nr] < self._limits[nr][0]:
                    self.rateconstants[nr] = self._limits[nr][0]
                    #if verbose: sys.stderr.write("DCPYPS: Warning: Corrected out-of-range rate constant\n")
                if self._rateconstants[nr] > self._limits[nr][1]:
                    self.rateconstants[nr] = self._limits[nr][1]
                    #if verbose: sys.stderr.write("DCPYPS: Warning: Corrected out-of-range rate constant\n")
    def _set_default_limits(self):
        if self._effectors[0] == 'c':
            self._limits = [[1e-15,1e+9]]
        else:
            self._limits = [[1e-15,1e+6]]

    def _set_constrain(self, is_constrained, func, args):
        self.is_constrained = is_constrained
        self.constrain_func = func
        self.constrain_args = args

class Graph(object):
    """
    Represents a kinetic mechanism as a graph.
    """
    def __init__(self, Rates):
        self.Rates = Rates
        self.graph = {}
        self.fixed = []
        self.constrained = []
        self.mr = []
        self.graph_from_rates(self.Rates)
        self.nodes, self.edges = self.nodes_edges(self.graph)

    def graph_from_rates(self, Rates):
        """
        Prepare a graph from given dcpyps mechanism rates. Graph is a
        dictionary of nodes. Each node has a list of connected states and a
        list of connection weights.
        """

        self.graph = {}
        for rate in Rates:
            w = 1
            if rate.fixed:
                self.fixed.append([rate.State1.name, rate.State2.name])
                w = 0
            if rate.is_constrained:
                self.constrained.append([rate.State1.name, rate.State2.name])
                w = 0
            if rate.mr:
                self.mr.append([rate.State1.name, rate.State2.name])
                w = 2

            if rate.State1.name not in self.graph:
                self.graph[rate.State1.name] = [[rate.State2.name], [w]]
            if rate.State1.name in self.graph:
                if rate.State2.name not in self.graph[rate.State1.name][0]:
                    self.graph[rate.State1.name][0].append(rate.State2.name)
                    self.graph[rate.State1.name][1].append(w)

    def nodes_edges(self, graph):
        """
        Prepare list of nodes and list of edges and their weights.
        """
        nodes = graph.keys()
        edges = [[],[]]
        for node in nodes:
            for next, w in zip(graph[node][0], graph[node][1]):
                edge1 = [node, next]
                edge2 = [next, node]
                if (edge1 not in edges[0]) and (edge2 not in edges[0]):
                    edges[0].append(edge1)
                    edges[1].append([w])
                elif edge1 in edges[0]:
                    edges[1][edges[0].index(edge1)].append(w)
                elif edge2 in edges[0]:
                    edges[1][edges[0].index(edge2)].append(w)
        return nodes, edges

    def graph_from_nodes(self, nodes, edges):
        """
        Prepare a graph (without weights) from given nodes and edges.
        """
        graph = {}
        for node in nodes:
            graph[node] = [[],[]]
            for edge, w in zip(edges[0], edges[1]):
                if node in edge:
                    graph[node][0].append(edge[edge.index(node)-1])
                    graph[node][1].append(w[edge.index(node)-1])
        return graph

    def adjacency(self, graph):
        """
        Construct adjacency matrix of a graph.
        """
        nodes = graph.keys()
        M = np.zeros((len(nodes), len(nodes)))
        for node in nodes:
            for adj in graph[node][0]:
                M[nodes.index(node), nodes.index(adj)] = 1
        return M

    def incidence(self, graph):
        """
        Construct incidence matrix of a graph.
        """
        nodes, edges = self.nodes_edges(graph)
        M = np.zeros((len(nodes), len(edges[0])))
        for node in nodes:
            for edge in edges[0]:
                if node in edge:
                    M[nodes.index(node), edges[0].index(edge)] = 1
        return M

    def degree(self, graph):
        """
        Construct degree matrix of a graph.
        """
        nodes, edges = self.nodes_edges(graph)
        M = np.zeros((len(nodes), len(nodes)))
        for node in nodes:
            for edge in edges[0]:
                if node in edge:
                    M[nodes.index(node), nodes.index(node)] += 1
        return M

    def degree_list(self, graph):
        """
        Construct degree list.
        """
        nodes, edges = self.nodes_edges(graph)
        L = [0] * len(nodes)
        for node in nodes:
            for edge in edges[0]:
                if node in edge:
                    L[nodes.index(node)] += 1
        return L

    def laplacian(self, graph):
        """
        Construct Laplacian matrix of a graph.
        """
        deg = self.degree(graph)
        adj = self.adjacency(graph)
        M = deg - adj
        return M

    def remove_node(self, nodes, edges, degrees):
        """
        Remove nodes with 1 or 0 edges.
        """
        nodes_to_remove = []
        for ind, dgr in enumerate(degrees):
            if dgr == 1 or dgr == 0:
                nodes_to_remove.append(nodes[ind])
                edges_to_remove = []
                for id, edge in enumerate(edges[0]):
                    if nodes[ind] in edge:
                        edges_to_remove.append(edge)
                for edge in edges_to_remove:
                    id = edges[0].index(edge)
                    edges[0].pop(id)
                    edges[1].pop(id)
        for node in nodes_to_remove:
            degrees.pop(nodes.index(node))
            nodes.pop(nodes.index(node))
        return nodes, edges, degrees

    def find_cycles(self, graph):
        """
        Find all cycles in a graph.
        """
        nodes, edges = self.nodes_edges(graph)
        degree = self.degree_list(graph)
        todo = nodes[:]
        todo_degree = degree[:]
        todo_edges = edges[:]
        cycles = []
        todo, todo_edges, todo_degree = self.remove_node(todo, todo_edges, todo_degree)
        while todo and len(todo) >= 4:
            new_graph = self.graph_from_nodes(todo, todo_edges)
            todo_degree = self.degree_list(new_graph)
            todo, todo_edges = self.nodes_edges(new_graph)
            cycle = self.find_cycle(new_graph)
            if cycle:
                cycles.append(cycle)
                for node in cycle:
                    todo_degree[todo.index(node)] = todo_degree[todo.index(node)] - 1
            todo, todo_edges, todo_degree = self.remove_node(todo, todo_edges, todo_degree)
        return cycles

    def find_cycle(self, graph):
        """
        Find if graph contains at least one cycle. Search stops when one cycle
        is found. Returns None if no cycle is found.
        """
        todo = set(graph.keys())
        while todo:
            node = todo.pop()
            stack = [node]
            while stack:
                top = stack[-1]
                for node in graph[top][0]:
                    if node in stack and node != stack[-2]:
                        return stack[stack.index(node):]
                    if node in todo:
                        stack.append(node)
                        todo.remove(node)
                        break
                else:
                    node = stack.pop()
        return None

class Mechanism:
    """
    Represents a complete kinetic mechanism (Q-matrix scheme).

    State types and their roles
    ---------------------------
    A  open states
    B  short-lived shut states (within burst)
    C  long-lived shut states (between bursts)
    D  desensitised states (between clusters)

    The Q matrix is partitioned into sub-matrices automatically after
    construction and whenever the effector concentration is changed via
    :meth:`set_eff`.

    Parameters
    ----------
    Rates : list[Rate]
        All transitions of the mechanism.
    Cycles : list[Cycle] or None
        Thermodynamic cycles (needed for microscopic-reversibility
        constraints).
    fastblock : bool
        Legacy parameter; use *fastKB* instead.
    fastKB : float or None
        Fast-blocker equilibrium constant (M); activates the fast-block
        correction in Popen calculations.
    mtitle : str
        Mechanism title.
    rtitle : str
        Rate-set title.
    verbose : bool
        Emit warnings to stderr.
    """

    def __init__(
        self,
        Rates: list[Rate],
        Cycles: list[Cycle] | None = None,
        fastblock: bool = False,
        fastKB: float | None = None,
        mtitle: str = '',
        rtitle: str = '',
        verbose: bool = False,
    ) -> None:
        # Fix: avoid shared mutable default for Cycles
        if Cycles is None:
            Cycles = []

        # set by set_EC50_constraint(); read by update_constrains()
        self._ec50_constraint = None

        self.Rates = Rates

        # TODO: construct cycles from Rates list
#        self.ncyc = ncyc   # number of cycles; could be deduced from the rates!
        self.Cycles = Cycles

        self.mtitle = mtitle
        self.rtitle = rtitle
        self.verbose = verbose

        self._set_fastKB(fastKB)

        # construct States end effectors from Rates:
        self.States = []
        # dictionary of effectors: {"name":concentration}
        self._effdict = {}
        for rate in self.Rates:
            if rate.State1 not in self.States:
                self.States.append(rate.State1)
            if rate.State2 not in self.States:
                self.States.append(rate.State2)
            # build up a dictionary of effectors and their values
            # according to the rates:
            for eff in rate.effectors:
                if eff not in self._effdict.keys() and eff is not None:
                    self._effdict[eff] = 1.0

        self.sort_states()
        self.set_Q()

    def _set_fastblock(self, fastblock):
        self._fastblock = fastblock
    def _get_fastblock(self):
        return self._fastblock
    fastblock = property(_get_fastblock, _set_fastblock)

    def _set_fastKB(self, fastKB):
        self._fastKB = fastKB
        self._fastblock = False
        if fastKB: self._fastblock = True
    def _get_fastKB(self):
        return self._fastKB
    fastKB = property(_get_fastKB, _set_fastKB)

#        self.fastblk = fastblk
#        self.KBlk = KBlk
#        self.set_Q()
        
    def sort_states(self):
        # REMIS: please check whether this makes sense
        # sort States according to state type:
        self.States.sort(key=lambda state: state.statetype)
        # assign Q matrix indices according to sorted list:
        for no, state in enumerate(self.States):
            state.no = no # ZERO-based!
            
        self.kA = 0
        self.kB = 0
        self.kC = 0
        self.kD = 0
        for State in self.States:
            if State.statetype=='A':
                self.kA += 1 # all open states
            if State.statetype=='B':
                self.kB += 1 # short lived shut (inside burst) shut states
            if State.statetype=='C':
                self.kC += 1 # long lived shut (between bursts) shut states
            if State.statetype=='D':
                self.kD += 1 # very long lived shut (between clusters)
        
        self.kE = self.kA + self.kB # burst states
        self.kF = self.kB + self.kC # intra and inter burst shut states
        self.kG = self.kA + self.kB + self.kC # cluster states
        self.kH = self.kC + self.kD # gap between clusters states
        self.kI = self.kB + self.kC + self.kD # all shut states
        self.k = self.kA + self.kB + self.kC + self.kD # all states

    def set_Q(self):

        self.Q = np.zeros((len(self.States), len(self.States)), dtype=np.float64)

        # Initialize all rates:
        for Rate in self.Rates:
            self.Q[Rate.State1.no, Rate.State2.no] = \
                Rate.unit_rate()

        # Update diagonal elements:
        for d in range(self.Q.shape[0]):
            self.Q[d,d] = 0
            self.Q[d,d] = -np.sum(self.Q[d])
            
        self.update_submat()

    def __repr__(self):
        #TODO: need nice table format
        str_repr = '\nclass dcpyps.Mechanism'
        str_repr += '\nValues of unit rates [1/sec]:'
        id = 0
        for rate in self.Rates:
            str_repr += ('\n' + str(id) + '\tFrom ' + rate.State1.name + '  \tto ' +
                         rate.State2.name + '    \t' + rate.name +
                         '   \t' + str(rate.unit_rate()))
            id += 1
                         
        for state in self.States:
            if state.statetype=='A':
                str_repr += ('\n\nConductance of state ' + state.name + ' (pS)  = ' +
                         '     {0:.5g}'.format(state.conductance * 1e12))

        str_repr += ('\n\nNumber of open states = {0:d}'.format(self.kA))
        str_repr += ('\nNumber of short-lived shut states (within burst) = {0:d}'
            .format(self.kB))
        str_repr += ('\nNumber of long-lived shut states (between bursts) = {0:d}'
            .format(self.kC))
        str_repr += ('\nNumber of desensitised states = {0:d}'.format(self.kD))

        str_repr += ('\n\nNumber of cycles = {0:d}'.format(len(self.Cycles)))
        for i in range(len(self.Cycles)):
            str_repr += ('\nCycle {0:d} is formed of states: '.format(i))
            for j in range(len(self.Cycles[i].states)):
                str_repr += (self.Cycles[i].states[j] + '  ')
            fprod, bprod = self.check_mr(self.Cycles[i])
            str_repr += ('\n\tforward product = {0:.9e}'.format(fprod))
            str_repr += ('\n\tbackward product = {0:.9e}'.format(bprod))

        return str_repr

    def printout(self, output=sys.stdout):
        #TODO: need nice table format
        output.write('%s' % self)

    def set_rateconstants(self, newrates):
        for nr, rate in enumerate(self.Rates):
            self.Rates[nr].rateconstants = newrates[nr]

    def unit_rates(self):
        return np.array([rate.unit_rate() for rate in self.Rates])

    def update_submat(self):
        for Rate in self.Rates:
            self.Q[Rate.State1.no, Rate.State2.no] = \
                Rate.calc(self._effdict)

        # Update diagonal elements
        for d in range(self.Q.shape[0]):
            self.Q[d,d] = 0
            self.Q[d,d] = -np.sum(self.Q[d])

#        self.eigenvals, self.A = qml.eigs(self.Q)
#        self.GAB, self.GBA = qml.iGs(self.Q, self.kA, self.kB)
        self.QFF = self.Q[self.kA:self.kG, self.kA:self.kG]
        self.QFA = self.Q[self.kA:self.kG, :self.kA]
        self.QAF = self.Q[:self.kA, self.kA:self.kG]
        self.QAA = self.Q[:self.kA, :self.kA]
        self.QEE = self.Q[:self.kE, :self.kE]
        self.QBB = self.Q[self.kA:self.kE, self.kA:self.kE]
        self.QAB = self.Q[:self.kA, self.kA:self.kE]
        self.QBA = self.Q[self.kA:self.kE, :self.kA]
        self.QBC = self.Q[self.kA:self.kE, self.kE:self.kG]
        self.QAC = self.Q[:self.kA, self.kE:self.kG]
        self.QCB = self.Q[self.kE:self.kG, self.kA:self.kE]
        self.QCA = self.Q[self.kE:self.kG, :self.kA]
        self.QII = self.Q[self.kA:self.k, self.kA:self.k]
        self.QIA = self.Q[self.kA:self.k, :self.kA]
        self.QAI = self.Q[:self.kA, self.kA:self.k]
        self.QGG = self.Q[:self.kG, :self.kG]

    def set_eff(self, eff, val):
        self.set_effdict({eff:val})

    def set_effdict(self, effdict):
        # check dictionary sanity:
        for effname, effvalue in effdict.items():
            if effname not in self._effdict.keys():
                if self.verbose:
                    sys.stderr.write("DCPYPS: Warning: None of the rates depends on effector %s\n" % effname)
#                errmsg = "DCPYPS: None of the rates depends on effector %s\n" % effname
#                raise RuntimeError(errmsg)
            else:
                self._effdict[effname] = effvalue
            
        self.update_submat()

    def theta(self):

        return np.array([rate.unit_rate() for rate in self.Rates
                         if rate.is_free])

    def get_free_parameter_names(self):

        return [rate.name for rate in self.Rates if rate.is_free]


    def theta_unsqueeze(self, theta):

        iter = 0
        for i in range(len(self.Rates)):
            if self.Rates[i].is_free:
                self.Rates[i].rateconstants = theta[iter]
                iter += 1
        self.update_constrains()
        self.update_mr()

    def update_states(self):
        """
        """

        self.sort_states()
        for i in range(len(self.Rates)):
            for state in self.States:
                if self.Rates[i].State1.name == state.name:
                    self.Rates[i].State1 = state
                if self.Rates[i].State2.name == state.name:
                    self.Rates[i].State2 = state

        self.set_Q()
        self.update_submat()
        self.update_constrains()
        self.update_mr()

    def _apply_rate_constraints(self):
        """Rates that are multiples of other rates, then microscopic reversibility.

        Split out of :meth:`update_constrains` so that the EC50 constraint can
        re-apply it for each trial value it tries, without recursing.
        """
        for i in range(len(self.Rates)):
            if self.Rates[i].is_constrained:
                args = self.Rates[i].constrain_args
                func = self.Rates[i].constrain_func
                self.Rates[i].rateconstants = func(self.Rates[args[0]].rateconstants, args[1])
        self.update_mr()

    def update_constrains(self):

        self._apply_rate_constraints()
        if self._ec50_constraint is not None:
            self._solve_EC50_constraint()

    def set_EC50_constraint(self, nrate, ec50, tres=0.0, eff='c', solver=None,
                            on_unreachable='clamp'):
        """Compute one rate constant from a specified equilibrium EC50.

        The option described by Colquhoun, Hatton & Hawkes (2003), p. 702: an
        EC50 measured independently is supplied, and the value of one rate
        constant is then calculated at each stage of a fit from that EC50 and
        the values of all the other rates. This reduces the number of free
        parameters by one, and is an alternative to fixing a rate at a guessed
        value.

        The constrained rate is excluded from :meth:`theta` and from
        :meth:`get_free_parameter_names`, and is recomputed by
        :meth:`update_constrains` -- which means on every call to
        :meth:`theta_unsqueeze`, i.e. at every iteration of a fit.

        Parameters
        ----------
        nrate : int
            Index of the rate to be computed.
        ec50 : float
            The equilibrium EC50 to impose [M].
        tres : float
            Resolution at which the EC50 is defined. Zero, the default, is the
            ideal equilibrium EC50.
        eff : str
            Effector whose EC50 this is.
        solver : callable, optional
            ``solver(mec) -> float``, returning the value the rate should take,
            or raising ``ArithmeticError`` if the EC50 cannot be reached. The
            default solves numerically, which costs one call to ``popen.EC50``
            per iteration and is therefore not cheap; a mechanism whose EC50
            can be inverted in closed form can supply one here instead.
        on_unreachable : {'clamp', 'raise'}
            What to do when no value of the rate gives the specified EC50 --
            which happens more readily than one might expect, because the EC50
            reachable by raising an association rate constant does not fall to
            zero but to a floor set by the other rates. ``'clamp'``, the
            default, puts the rate at whichever of its limits is nearest to
            satisfying the constraint and counts the event in
            :attr:`ec50_clamped`, so that the fit continues and the likelihood
            can move the other rates until the EC50 becomes reachable. This is
            how the paper treats a rate that runs into a limit (p. 702).
            ``'raise'`` raises ``ArithmeticError`` instead.

        Notes
        -----
        The numerical default assumes the EC50 falls as the rate rises, which
        holds for an association rate constant, and brackets the root by
        expanding from the rate's current value.
        """
        if not 0 <= nrate < len(self.Rates):
            raise IndexError(f"no rate with index {nrate}")
        if not ec50 > 0:
            raise ValueError("EC50 must be positive")
        if on_unreachable not in ('clamp', 'raise'):
            raise ValueError("on_unreachable must be 'clamp' or 'raise'")
        self.Rates[nrate].ec50_constrained = True
        self._ec50_constraint = dict(nrate=nrate, ec50=float(ec50),
                                     tres=float(tres), eff=eff, solver=solver,
                                     on_unreachable=on_unreachable, clamped=0)
        self.update_constrains()

    @property
    def ec50_clamped(self):
        """How often the EC50 constraint has been unreachable and clamped."""
        if self._ec50_constraint is None:
            return 0
        return self._ec50_constraint['clamped']

    def clear_EC50_constraint(self):
        """Remove an EC50 constraint, making the rate free again."""
        if self._ec50_constraint is not None:
            self.Rates[self._ec50_constraint['nrate']].ec50_constrained = False
            self._ec50_constraint = None

    def _solve_EC50_constraint(self):
        from scalcs import popen      # local: popen imports scalcslib

        c = self._ec50_constraint
        nrate, target, tres = c['nrate'], c['ec50'], c['tres']
        rate = self.Rates[nrate]
        # limits are normalised to [[lower, upper]] by Rate._check_limits
        low, high = rate.limits[0] if rate.limits else (1e-4, 1e14)

        def settle(value):
            rate.rateconstants = float(value)
            self._apply_rate_constraints()

        def unreachable(nearest):
            """The EC50 cannot be had at any allowed value of this rate."""
            if c['on_unreachable'] == 'raise':
                raise ArithmeticError(
                    "cannot reach an EC50 of {0:g} M by changing {1} within "
                    "[{2:g}, {3:g}]".format(target, rate.name, low, high))
            c['clamped'] += 1
            settle(nearest)

        if c['solver'] is not None:
            try:
                value = float(c['solver'](self))
            except ArithmeticError:
                # a closed-form solver signals "no positive root" this way;
                # more affinity is always the direction that lowers the EC50
                unreachable(high)
                return
            if low <= value <= high:
                settle(value)
            else:
                unreachable(high if value > high else low)
            return

        # popen.EC50 moves the effector; put it back when we are done
        saved = self._effdict.get(c['eff'])

        def mismatch(logk):
            settle(math.exp(logk))
            found = popen.EC50(self, tres, eff=c['eff'])
            if not (found > 0) or not math.isfinite(found):
                raise ArithmeticError(
                    "no EC50 for this mechanism at rate {0:g}".format(
                        math.exp(logk)))
            return math.log(found) - math.log(target)

        def restore():
            if saved is not None:
                self._effdict[c['eff']] = saved
                self.update_submat()

        loglow, loghigh = math.log(low), math.log(high)
        x0 = min(max(math.log(float(np.ravel(rate.rateconstants)[0])),
                     loglow), loghigh)
        xlo = xhi = x0
        flo = fhi = mismatch(x0)
        # the EC50 falls as an association rate rises, so raising the rate
        # lowers the mismatch; expand whichever end has not yet crossed zero
        while flo > 0 and fhi > 0 and xhi < loghigh:
            xhi = min(xhi + math.log(10.0), loghigh)
            fhi = mismatch(xhi)
        while flo < 0 and fhi < 0 and xlo > loglow:
            xlo = max(xlo - math.log(10.0), loglow)
            flo = mismatch(xlo)

        if flo * fhi > 0:
            unreachable(high if fhi > 0 else low)
            restore()
            return

        root = so.brentq(mismatch, xlo, xhi, xtol=1e-12, rtol=1e-14)
        settle(math.exp(root))
        restore()

    def set_mr(self, mr, nrate, ncycle=0):
        """
        """

        self.Rates[nrate].mr = mr # redundant
        if mr:
            if ((self.Rates[nrate].State1.name in self.Cycles[ncycle].states) and
                (self.Rates[nrate].State2.name in self.Cycles[ncycle].states)):
                    #if mr:
                    self.Cycles[ncycle].mrconstr = [self.Rates[nrate].State1.name, self.Rates[nrate].State2.name]
                    #else:
                    #    self.Cycles[ncycle].mrconstr = []
            else:
                sys.stderr.write("DCPYPS: Warning: MR1: Proposed rate to be constrained is not in the cycle.")
        self.update_mr()

    def update_mr(self):
        #TODO: check for consistency between cycle.mrconstr and rate.mr.

        ids = []
        for cycle in self.Cycles:
            if cycle.mrconstr:
                # check if constrain is correct: should be just one rate per cycle
                # and between two connected states which are in cycle.
                if len(cycle.mrconstr) != 2:
                    sys.stderr.write("DCPYPS: Warning: MR: Only rate between TWO neighboring states can be constrained.")

                states1 = cycle.states[:]
                states2 = cycle.states[:]
                states2.append(states2.pop(0))
                exist = False
                for i in range(len(states1)):
                    if (((cycle.mrconstr[0] == states1[i]) and
                            (cycle.mrconstr[1] == states2[i])) or
                        ((cycle.mrconstr[1] == states1[i]) and
                            (cycle.mrconstr[0] == states2[i]))):
                                exist = True

                if not exist:
                    sys.stderr.write("DCPYPS: Warning: MR2: Proposed rate to be constrained is not in the cycle.")

                prod = 1
                id = None
                forward = False
                for j in range(len(states1)):
                    icount = 0
                    for rate in self.Rates:
                        if ((states1[j] == rate.State1.name) and
                                (states2[j] == rate.State2.name)):
                            if ((cycle.mrconstr[0] == states1[j]) and
                                (cycle.mrconstr[1] == states2[j])):
                                id = icount
                                forward = True
                            else:
                                prod = prod * rate.rateconstants
                        if ((states1[j] == rate.State2.name) and
                                (states2[j] == rate.State1.name)):
                            if ((cycle.mrconstr[1] == states1[j]) and
                                (cycle.mrconstr[0] == states2[j])):
                                id = icount
                            else:
                                prod = prod / rate.rateconstants
                        icount += 1
                if forward:
                    prod = 1 / prod
                self.Rates[id].rateconstants = prod
                ids.append(id)

        return ids

    def check_mr(self, cycle):

        fprod = 1.0
        bprod = 1.0
        states1 = cycle.states[:]
        states2 = cycle.states[:]
        states2.append(states2.pop(0))

        for j in range(len(states1)):
            for rate in self.Rates:
                if ((states1[j] == rate.State1.name) and
                        (states2[j] == rate.State2.name)):
                    fprod = fprod * rate.rateconstants
                if ((states1[j] == rate.State2.name) and
                        (states2[j] == rate.State1.name)):
                    bprod *= rate.rateconstants

        return fprod[0], bprod[0]
    
    def check_limits(self):
        """
        Check if current rate constants are within specified limits.
        Returns 'True' if all rate constants are within the specified limits. 
        Returns 'False' if any of rate constants is outside the specified
        limits.
        """
        
        withinLimits = True
        for i in range(len(self.Rates)):
            if ((self.Rates[i].unit_rate() < self.Rates[i].limits[0][0]) or
                (self.Rates[i].unit_rate() > self.Rates[i].limits[0][1])):
                withinLimits = False
        return withinLimits
    
    def impose_limits(self):
        """
        Check if current rate constants are within specified limits and 
        set limits if rate constant is outside the specified limits.
        """
        
        for i in range(len(self.Rates)):
            if ((self.Rates[i].unit_rate() < self.Rates[i].limits[0][0]) or
                (self.Rates[i].unit_rate() > self.Rates[i].limits[0][1])):
                self.Rates[i]._impose_limits()

