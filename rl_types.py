import collections
import typing

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

Shape = typing.Tuple[int]
