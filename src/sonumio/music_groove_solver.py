"""Exact bounded grid rhythm optimization for a monophonic drum track.

A state is the number of selected slots, the first slot and the most
recent slot. Costs are integer-valued; tie breaks are lexicographic.
Mandatory/forbidden hits, minimum/maximum gaps, wraparound and exact
hit counts are hard constraints, not postprocessing heuristics.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

from .errors import GenerationError
from .arrangement import integer

SLOTS=16
MAX_TRANSITIONS=2_000_000


@dataclass(frozen=True,slots=True)
class GrooveConstraints:
    hits:int
    minimum_gap:int=1
    maximum_gap:int=16
    mandatory:tuple[int,...]=()
    forbidden:tuple[int,...]=()
    cyclic:bool=True

    def __post_init__(self):
        integer(self.hits,0,SLOTS,"groove hit count")
        integer(self.minimum_gap,1,SLOTS,"groove minimum gap")
        integer(self.maximum_gap,self.minimum_gap,SLOTS,"groove maximum gap")
        for name in ("mandatory","forbidden"):
            slots=getattr(self,name)
            if (type(slots) is not tuple or any(type(s) is not int or not 0<=s<SLOTS for s in slots)
                    or slots!=tuple(sorted(set(slots)))):
                raise GenerationError("Groove slot constraints must be unique and increasing")
        if set(self.mandatory)&set(self.forbidden):
            raise GenerationError("Groove mandatory and forbidden slots overlap")
        if len(self.mandatory)>self.hits or self.hits>SLOTS-len(self.forbidden):
            raise GenerationError("Groove hit count cannot satisfy slot constraints")
        if type(self.cyclic) is not bool:
            raise GenerationError("Groove cyclic switch must be boolean")


@dataclass(frozen=True,slots=True)
class GrooveObjective:
    slot_costs:tuple[int,...]
    gap_weight:int=1

    def __post_init__(self):
        if (type(self.slot_costs) is not tuple or len(self.slot_costs)!=SLOTS
                or any(type(v) is not int or not -4096<=v<=4096 for v in self.slot_costs)):
            raise GenerationError("Invalid groove slot costs")
        integer(self.gap_weight,0,32,"groove gap weight")


def _gap_cost(gap,hits,objective):
    # Rational ideal spacing 16/hits. Multiplication by hits avoids
    # floating-point ties and rewards evenly spaced hits without forcing it.
    return objective.gap_weight*abs(gap*hits-SLOTS)


def _legal_gap(a,b,constraints):
    return constraints.minimum_gap<=b-a<=constraints.maximum_gap


def _skipped_mandatory(a,b,mandatory):
    return any(a<s<b for s in mandatory)


def validate_pattern(pattern,constraints):
    if not isinstance(constraints,GrooveConstraints):
        raise GenerationError("Expected groove constraints")
    if (type(pattern) is not tuple or len(pattern)!=constraints.hits
            or any(type(s) is not int or not 0<=s<SLOTS for s in pattern)
            or pattern!=tuple(sorted(set(pattern)))):
        raise GenerationError("Invalid groove hit pattern")
    if not set(constraints.mandatory)<=set(pattern) or set(constraints.forbidden)&set(pattern):
        raise GenerationError("Groove pattern violates required slots")
    if any(not _legal_gap(a,b,constraints) for a,b in zip(pattern,pattern[1:])):
        raise GenerationError("Groove pattern violates internal gap bounds")
    if pattern and constraints.cyclic and not _legal_gap(pattern[-1],pattern[0]+SLOTS,constraints):
        raise GenerationError("Groove pattern violates cyclic gap bounds")
    return pattern


def pattern_cost(pattern,constraints,objective):
    if not isinstance(objective,GrooveObjective):
        raise GenerationError("Expected groove objective")
    validate_pattern(pattern,constraints)
    cost=sum(objective.slot_costs[s] for s in pattern)
    cost+=sum(_gap_cost(b-a,constraints.hits,objective) for a,b in zip(pattern,pattern[1:]))
    if pattern and constraints.cyclic:
        cost+=_gap_cost(pattern[0]+SLOTS-pattern[-1],constraints.hits,objective)
    return cost


def solve_groove(constraints,objective):
    """Return the exact optimum and explicit search-work receipt."""
    if not isinstance(constraints,GrooveConstraints) or not isinstance(objective,GrooveObjective):
        raise GenerationError("Expected groove constraints and objective")
    if not constraints.hits:
        return (),{"cost":0,"transitions":0,"feasible":True,"optimal":True}
    available=tuple(s for s in range(SLOTS) if s not in constraints.forbidden)
    mandatory=constraints.mandatory
    best=None
    transitions=0
    for first in available:
        if mandatory and first>mandatory[0]:
            continue
        states={first:(objective.slot_costs[first],(first,))}
        for count in range(1,constraints.hits):
            next_states={}
            for last,(cost,path) in states.items():
                for slot in available:
                    if slot<=last:
                        continue
                    transitions+=1
                    if transitions>MAX_TRANSITIONS:
                        raise GenerationError("Groove transition budget exceeded")
                    if not _legal_gap(last,slot,constraints) or _skipped_mandatory(last,slot,mandatory):
                        continue
                    if sum(s>slot for s in available)<constraints.hits-count-1:
                        continue
                    candidate=(cost+objective.slot_costs[slot]
                               +_gap_cost(slot-last,constraints.hits,objective),path+(slot,))
                    if slot not in next_states or candidate<next_states[slot]:
                        next_states[slot]=candidate
            states=next_states
            if not states:
                break
        for last,(cost,path) in states.items():
            if len(path)!=constraints.hits or (mandatory and last<mandatory[-1]):
                continue
            if constraints.cyclic:
                if not _legal_gap(last,first+SLOTS,constraints):
                    continue
                cost+=_gap_cost(first+SLOTS-last,constraints.hits,objective)
            candidate=(cost,path)
            if best is None or candidate<best:
                best=candidate
    if best is None:
        raise GenerationError("No feasible groove under declared hard constraints")
    cost,path=best
    if pattern_cost(path,constraints,objective)!=cost:
        raise GenerationError("Groove optimum receipt mismatch")
    return path,{"cost":cost,"transitions":transitions,"feasible":True,"optimal":True}
