Exploring physics informed neural networks. The work here follows the [original
paper](https://arxiv.org/abs/1711.10561). An accompanying blog post can be found
on [my website](https://mckuzyk.com/posts/physics-informed-neural-nets/). Future
work, data driven discovery of nonlinear partial differential equations (the
inverse problem), following [part two](https://arxiv.org/pdf/1711.10566) of the
original paper.

To get started with this code:
```
git clone https://github.com/mckuzyk/pinn_heat.git
cd pinn_heat
uv run pinn_heat/run.py --experiment test
```

See `pinn_heat/experiments.py` for the current set of parameters that can be
run, or define your own!
