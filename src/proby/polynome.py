
from dataclasses import dataclass
from itertools import chain
from typing import Iterable
import numpy as np


@dataclass(frozen=True, eq=True)
class SymbolicVariable:
    id: str
    def __le__(self, other: "SymbolicVariable") -> bool:
        return self.id <= other.id

    def __str__(self) -> str:
        return self.id

@dataclass(frozen=True, eq=True)
class Monad:
    sym: SymbolicVariable
    power: int

    def __str__(self) -> str:
        return f"{str(self.sym)}^{self.power}"

    def call(self, **kwargs):
        if self.sym.id not in kwargs:
            raise ValueError(f"Need sym {self.sym}. Not found in kwargs: {kwargs}")
        return kwargs[self.sym.id] ** self.power


def next_none(x):
    try:
        return next(x)
    except StopIteration:
        return None


def merge_dict_sorted(x: Iterable[Monad], y: Iterable[Monad]) -> list[Monad]:
    x_it = iter(x)
    y_it = iter(y)
    xval = next_none(x_it)
    yval = next_none(y_it)
    out: list[Monad] = []
    def append_stack(val: Monad):
        if not out or out[-1].sym != val.sym:
            out.append(val)
        else:
            out.append(Monad(sym=val.sym, power=val.power + out.pop().power))
        
    while not (xval is None and yval is None):
        if yval is None or (xval is not None and xval.sym <= yval.sym):
            append_stack(xval)
            xval = next_none(x_it)
        else:
            append_stack(yval)
            yval = next_none(y_it)
    return out


@dataclass(frozen=True, eq=True)
class Monome:
    coefficient: int
    monads: tuple[Monad, ...]

    def __mul__(self, other: "Monome") -> "Monome":
        return Monome(monads=tuple(merge_dict_sorted(self.monads, other.monads)), coefficient=self.coefficient*other.coefficient)

    def __str__(self) -> str:
        return f"{self.coefficient:+d}" + "".join([str(monad) for monad in self.monads])
    
    def call(self, **kwargs):
        return self.coefficient * np.prod(np.vstack([monad.call(**kwargs) for monad in self.monads]), axis=0)

    def get_sym_vars(self) -> set[SymbolicVariable]:
        return set(monad.sym for monad in self.monads)

@dataclass(frozen=True, eq=True)
class Polynome:
    monomes: tuple[Monome]

    def dot(self, other: Monome) -> "Polynome":
        return Polynome(monomes=tuple(monome * other for monome in self.monomes))

    def __add__(self, other: "Polynome") -> "Polynome":
        monome_kdx: dict[tuple[Monad, ...], int] = {}
        for monome in chain(self.monomes, other.monomes):
            monome_kdx.setdefault(monome.monads, 0)
            monome_kdx[monome.monads] += monome.coefficient
        return Polynome(monomes=tuple(Monome(coefficient=v, monads=k) for k, v in monome_kdx.items()))

    def __radd__(self, other: "Polynome") -> "Polynome":
        if other == 0:
            return self
        else:
            return other + self

    def __mul__(self, other: "Polynome") -> "Polynome":
        return sum([self.dot(o_mon) for o_mon in other.monomes])

    def __str__(self) -> str:
        return " ".join([str(monome) for monome in self.monomes])
    
    def call(self, **kwargs):
        return sum(monome.call(**kwargs) for monome in self.monomes)

    def get_sym_vars(self) -> set[SymbolicVariable]:
        sym_vars_sets = [monome.get_sym_vars() for monome in self.monomes]
        return set.union(*sym_vars_sets)
