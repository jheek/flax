# Lint as: python3
"""TODO(jheek): DO NOT SUBMIT without one-line documentation for scope.

TODO(jheek): DO NOT SUBMIT without a detailed description of scope.
"""

import enum
import functools
import hashlib
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, TypeVar, Union

from . import tracers
from .frozen_dict import freeze
from .frozen_dict import FrozenDict
from .frozen_dict import unfreeze

import jax
from jax import lax
from jax import random


T = TypeVar('T')


PRNGKey = Any
Array = Any


KindFilter = Union[bool, str, Sequence[str]]

MaybeFrozenKind = Union[Dict[str, Any], FrozenDict[str, Any]]

Variables = Dict[str, MaybeFrozenKind]

def _fold_in_str(rng: PRNGKey, data: str) -> PRNGKey:
  """Fold a string into a jax.random.PRNGKey using its SHA-1 hash."""
  m = hashlib.sha1()
  m.update(data.encode('utf-8'))
  d = m.digest()
  hash_int = int.from_bytes(d[:4], byteorder='big')
  return random.fold_in(rng, hash_int)


def in_kind_filter(kind_filter: KindFilter, kind: str) -> bool:
  if isinstance(kind_filter, str):
    return kind == kind_filter
  if isinstance(kind_filter, Sequence) and not isinstance(kind_filter, str):
    return kind in kind_filter
  if isinstance(kind_filter, bool):
    return kind_filter
  raise TypeError('Invalid KindFilter')



class Scope:
  """Scope."""

  def __init__(self,
               variables: Variables,
               rngs: Optional[Dict[str, PRNGKey]] = None,
               name: Optional[str] = None,
               parent: Optional['Scope'] = None):
    # TODO: Use EasyDict here
    self.parent = parent
    self.name = name
    self.variables = variables
    self.rngs = rngs if rngs else {}

    self.trace_level = tracers.trace_level(tracers.current_trace())
    
    self.rng_counters = {key: 0 for key in self.rngs}
    self._populate_kinds()  # QUESTION: Is this a performance hit?

  def _validate_trace_level(self):
    tracers.check_trace_level(self.trace_level)

  def push(self, name: str) -> 'Scope':
    self._validate_trace_level()
    rngs = {key: _fold_in_str(rng, name) for key, rng in self.rngs.items()}
    scope = Scope(variables={}, name=name, rngs=rngs, parent=self)
    return scope


  def fold_rngs(self, data: str):
    self._validate_trace_level()  # QUESTION: Should this be here?
    return {key: _fold_in_str(rng, data) for key, rng in self.rngs.items()}
      
  def get_kind(self, kind: str, mutable: bool = False) -> MaybeFrozenKind:
    """Returns all variable of a given kind."""
    if kind not in self.variables:
      if self.parent:
        parent_kind = self.parent.get_kind(kind, mutable)
        if self.name not in parent_kind:
          if isinstance(parent_kind, FrozenDict):
            return FrozenDict()
          parent_kind[self.name] = {}
        self.variables[kind] = parent_kind[self.name]
      elif mutable:
        self.variables[kind] = {}
      else:
        return FrozenDict()
    return self.variables[kind]

  def has_rng(self, kind: str) -> bool:
    return kind in self.rngs

  def make_rng(self, kind: str) -> PRNGKey:
    if not self.has_rng(kind):
      raise ValueError('Need a rng for kind \'{}\''.format(kind))
    self._validate_trace_level()
    self.rng_counters[kind] += 1
    return random.fold_in(self.rngs[kind], self.rng_counters[kind])

  def get_variable(self, kind: str, name: str, default: T = None) -> T:
    variables = self.get_kind(kind)
    if name in variables:
      return variables[name]
    else:
      # QUESTION: Should this put the default in, and then remove that
      # logic from `param?`? Why should we have a `param` method anyways?
      return default

  def has_variable(self, kind: str, name: str) -> bool:
    variables = self.get_kind(kind)
    return name in variables

  def put_variable(self, kind: str, name: str, value: Any):
    self._validate_trace_level()
    variables = self.get_kind(kind)
    variables[name] = value

  def param(self, name: str, init_fn: Callable[..., T], *init_args) -> T:
    if not self.has_variable('param', name):
      init_value = init_fn(self.make_rng('param'), *init_args)
      self.put_variable('param', name, init_value)
    return self.get_variable('param', name)

  def _populate_kinds(self):
    if self.parent:
      kinds = self.parent.variables.keys()
      for kind in kinds:
        self.get_kind(kind)


def _unfreeze_variables(variables, mutable):
  new_variables = {}
  for key, value in variables.items():
    if in_kind_filter(mutable, key):
      new_variables[key] = unfreeze(value)
    else:
      new_variables[key] = value
  return new_variables

