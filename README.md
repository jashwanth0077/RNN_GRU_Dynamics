# RNN_GRU_Dynamics
This assignment is inspired by [https://proceedings.mlr.press/v28/pascanu13.pdf](Pascanu et al’s paper) on the learning dynamics of RNNs. Here,
you will:
1. Implement a vanilla RNN and a GRU from the equations (no torch.nn.RNN/GRU).
2. Train them on synthetic long-range dependency tasks.
3. Log and interpret diagnostics of the gradients, and relate them to activation/gate saturation,
and possible gradient vanishing and exploding.
4. Compare the behavior of RNN vs GRU under different regimes (with/without clipping; tanh
vs sigmoid).
