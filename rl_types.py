import collections
import typing

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'discounts', 'next_observations'])

Shape = typing.Tuple[int]
