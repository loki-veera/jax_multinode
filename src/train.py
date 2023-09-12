"""Multinode training in JAX."""

import os
from copy import deepcopy
from functools import partial
from typing import Tuple

import flax.linen as nn
import jax
import jax.experimental.multihost_utils as jax_mhu
import jax.numpy as jnp
import optax

# import builtins
from flax.core.frozen_dict import FrozenDict

# Settings for MultiNode setup
node_id = os.environ["SLURM_NODEID"]
visible_devices = [int(gpu) for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]

# # Override the print function to print only on the zeroth node, if necessary
# def print(*args, **kwargs):
#     if node_id == '0':
#         builtins.print(*args, **kwargs)


gpus_avail = len(
    visible_devices
)  # This represents the number of gpus required per node.


class Model(nn.Module):
    """Model defintion."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x (jnp.ndarray): Input (BATCH_SIZE X 64)

        Returns:
            jnp.ndarray: Network approximation (BATCH_SIZE X 64)
        """
        x = nn.gelu(nn.Dense(128)(x))
        x = nn.gelu(nn.Dense(256)(x))
        x = nn.gelu(nn.Dense(512)(x))
        x = nn.gelu(nn.Dense(1024)(x))
        x = nn.gelu(nn.Dense(1024)(x))
        x = nn.gelu(nn.Dense(512)(x))
        x = nn.gelu(nn.Dense(256)(x))
        x = nn.gelu(nn.Dense(128)(x))
        x = nn.Dense(64)(x)
        return x


@partial(jax.jit, static_argnames=["model"])
def forward_pass(
    net_state: FrozenDict,
    model: nn.Module,
    batch_input: jnp.ndarray,
    batch_output: jnp.ndarray,
) -> jnp.ndarray:
    """Compute forward pass and MSE loss.

    Args:
        net_state (FrozenDict): Model Parameters
        model (nn.Module): Model instance
        batch_input (jnp.ndarray): Input batch
        batch_output (jnp.ndarray): Output batch

    Returns:
        jnp.ndarray: MSE Loss
    """
    preds = model.apply(net_state, x=batch_input)
    mse_cost = jnp.mean(0.5 * (preds - batch_output) ** 2)
    return mse_cost


val_grad_fn = jax.value_and_grad(forward_pass, argnums=0)


def train_step(
    net_state: FrozenDict,
    model: nn.Module,
    batch_input: jnp.ndarray,
    batch_output: jnp.ndarray,
) -> Tuple[float, FrozenDict]:
    """Train step to extract gradients from all the hosts.

    Args:
        net_state (FrozenDict): Model Parameters
        model (nn.Module): Model instance
        batch_input (jnp.ndarray): Input batch
        batch_output (jnp.ndarray): Output batch

    Returns:
        Tuple[float, FrozenDict]: Tuple containing the loss and gradients from all the devices
    """
    loss, grads = val_grad_fn(net_state, model, batch_input, batch_output)
    grads = jax.lax.pmean(grads, axis_name="i")
    loss = jax.lax.pmean(loss, axis_name="i")
    return loss, grads


split_fn = lambda x: jnp.stack(jnp.split(x, gpus_avail, axis=0), axis=0)


def step(
    net_state: FrozenDict,
    model: nn.Module,
    input: jnp.ndarray,
    output: jnp.ndarray,
    opt_state: FrozenDict,
    opt: optax.GradientTransformation,
) -> Tuple[float, FrozenDict, FrozenDict]:
    """Step function to apply the gradients.

    Args:
        net_state (FrozenDict): Model parameters
        model (nn.Module): Model instance
        input (jnp.ndarray): Input batch
        output (jnp.ndarray): Output batch
        opt_state (FrozenDict): Optimizer state
        opt (optax.GradientTransformation): Optimizer

    Returns:
        Tuple[float, FrozenDict, FrozenDict]: Tuple containing loss, model parameters and optimizer state
    """
    # Wait till all the process complete the job
    jax_mhu.sync_global_devices(name="let_all_sync")
    partial_diff = partial(train_step, net_state=net_state, model=model)
    loss, grads = jax.pmap(
        partial_diff, devices=jax.local_devices()[:gpus_avail], axis_name="i"
    )(batch_input=split_fn(input), batch_output=split_fn(output))
    loss = jnp.mean(loss)
    grads = jax.tree_map(partial(jnp.mean, axis=0), grads)
    (
        updates,
        opt_state,
    ) = opt.update(grads, opt_state, net_state)
    net_state = optax.apply_updates(net_state, updates)
    return loss, net_state, opt_state


def train():
    """Training function."""
    global gpus_avail
    key = jax.random.PRNGKey(0)
    model = Model()
    net_state = model.init(key, jnp.ones((1, 64)))
    opt = optax.adam(learning_rate=1e-4)
    opt_state = opt.init(net_state)
    iters = 10
    epochs = 10
    bs = 2048
    for e in range(epochs):
        loss_iter = 0.0
        for _ in range(iters):
            key = jax.random.PRNGKey(epochs + iters)
            batch_input = jnp.array(jax.random.normal(key, shape=(bs, 64)))
            batch_output = deepcopy(batch_input)
            batch_output = batch_output + 1.0
            loss, net_state, opt_state = step(
                net_state, model, batch_input, batch_output, opt_state, opt
            )
            loss_iter += loss
        print(
            f"Epoch: {e} on Node: {node_id} --> Loss: {loss_iter / iters}", flush=True
        )


if __name__ == "__main__":
    jax.distributed.initialize(local_device_ids=visible_devices)
    train()
    jax.distributed.shutdown()
