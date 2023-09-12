import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial
from copy import deepcopy

gpus_avail = 1

class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
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

@partial(jax.jit, static_argnames=['model'])
def forward_pass(net_state, model, batch_input, batch_output):
    preds = model.apply(
        net_state,
        x = batch_input
    )
    mse_cost = jnp.mean(0.5*(preds - batch_output)**2)
    return mse_cost

val_grad_fn = jax.value_and_grad(forward_pass, argnums=0)
def train_step(net_state, model, batch_input, batch_output):
    loss, grads = val_grad_fn(net_state, model, batch_input, batch_output)
    return loss, grads

def step(net_state, model, input, output, opt_state, opt):
    split_fn = lambda x: jnp.stack(jnp.split(x, gpus_avail, axis=0), axis=0)
    partial_diff = partial(
        train_step,
        net_state = net_state,
        model = model
    )
    loss, grads = jax.pmap(
        partial_diff,
        devices=jax.devices()[:gpus_avail],
    )(batch_input=split_fn(input), batch_output=split_fn(output))
    loss = jnp.mean(loss)
    grads = jax.tree_map(partial(jnp.mean, axis=0), grads)
    updates, opt_state, = opt.update(grads, opt_state, net_state)
    net_state = optax.apply_updates(net_state, updates)
    return loss, net_state, opt_state

def train():
    global gpus_avail
    key = jax.random.PRNGKey(0)
    model = Model()
    net_state = model.init(
        key,
        jnp.ones((1, 64))
    )
    opt = optax.adam(learning_rate=1e-4)
    opt_state = opt.init(net_state)
    iters = 10
    epochs = 10
    bs = 32
    for e in range(epochs):
        loss_iter = 0.0
        for i in range(iters):
            key = jax.random.PRNGKey(epochs+iters)
            batch_input = jnp.array(jax.random.normal(key, shape=(bs, 64)))
            batch_output = deepcopy(batch_input)
            batch_output = batch_output + 1.
            loss, net_state, opt_state = step(net_state, model, batch_input, batch_output, opt_state, opt)
            loss_iter += loss
        print(f"Epoch: {e} --> Loss: {loss_iter/iters}")

if __name__ == '__main__':
    train()
