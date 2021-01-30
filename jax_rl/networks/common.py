import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.optim.base import Optimizer

default_init = nn.initializers.orthogonal
PRNGKey = Any
Params = flax.core.FrozenDict
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activate_final: int = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.relu(x)
        return x


class Parameter(nn.Module):
    shape: Shape
    init: Callable[[PRNGKey, Shape, Dtype],
                   jnp.ndarray] = nn.initializers.zeros

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        return self.param('parameter', self.init, self.shape)


@flax.struct.dataclass
class Model:
    fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Optional[Params] = None
    optimizer: Optional[Optimizer] = None

    def with_optimizer(self, optim_def):
        optimizer = optim_def.create(self.params)
        return self.replace(params=None, optimizer=optimizer)

    def __call__(self, *args, **kwargs):
        return self.fn.apply({'params': self.params or self.optimizer.target},
                             *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad, info = grad_fn(self.optimizer.target)
        new_optimizer = self.optimizer.apply_gradient(grad)
        return self.replace(optimizer=new_optimizer), info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            params = self.params or self.optimizer.target
            f.write(flax.serialization.to_bytes(params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = self.params or self.optimizer.target
            params = flax.serialization.from_bytes(params, f.read())
        return self.replace(params=params, optimizer=None)


def create_model(model_def: nn.Module, inputs: Sequence[jnp.ndarray]) -> Model:
    params = model_def.init(*inputs)['params']
    return Model(model_def, params=params)
