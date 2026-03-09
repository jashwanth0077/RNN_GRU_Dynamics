
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# DO THIS
def spectral_radius(mat: np.ndarray) -> float:
    # The spectral radius of a matrix is the largest absolute value of its eigenvalues.
    # mat is (H,H)
    if mat.size == 0: return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(mat))))


# DO THIS
def _tanh_saturation_distance(h: torch.Tensor) -> torch.Tensor:
    """Distance to saturation for tanh outputs in [-1, 1].
    """
    return 1.0 - torch.abs(h)


# DO THIS
def _sigmoid_saturation_distance(h: torch.Tensor) -> torch.Tensor:
    """Distance to saturation for sigmoid outputs in [0, 1].
    """
    return torch.min(h, 1.0 - h)
    


class VanillaRNN(nn.Module):
    """
    Pascanu-style vanilla RNN:
      h_t = act(h_{t-1} W_hh + u_t W_uh + b_hh)
    Heads:
      - lastSoftmax: softmax on last hidden
      - softmax: softmax at every step
      - lastLinear: linear regression on last hidden
    """
    def __init__(
        self,
        nin: int,
        nout: int,
        nhid: int,
        init: str = "smart_tanh",
        classif_type: str = "lastSoftmax",
        rng=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.nhid = nhid
        self.init = init
        self.classif_type = classif_type

        if init == "sigmoid":
            W_uh = rng.normal(loc=0.0, scale=0.01, size=(nin, nhid)).astype(np.float32)
            W_hh = rng.normal(loc=0.0, scale=0.01, size=(nhid, nhid)).astype(np.float32)
            W_hy = rng.normal(loc=0.0, scale=0.01, size=(nhid, nout)).astype(np.float32)
            b_hh = np.zeros((nhid,), dtype=np.float32)
            b_hy = np.zeros((nout,), dtype=np.float32)
            self.act_name = "sigmoid"
        elif init == "test":
            W_uh = rng.normal(loc=0.0, scale=0.8, size=(nin, nhid)).astype(np.float32)
            W_hh = rng.normal(loc=0.0, scale=0.8, size=(nhid, nhid)).astype(np.float32)
            W_hy = rng.normal(loc=0.0, scale=0.8, size=(nhid, nout)).astype(np.float32)
            b_hh = np.zeros((nhid,), dtype=np.float32)
            b_hy = np.zeros((nout,), dtype=np.float32)
            self.act_name = "identity"
        elif init == "basic_tanh":
            W_uh = rng.normal(loc=0.0, scale=0.1, size=(nin, nhid)).astype(np.float32)
            W_hh = rng.normal(loc=0.0, scale=0.1, size=(nhid, nhid)).astype(np.float32)
            W_hy = rng.normal(loc=0.0, scale=0.1, size=(nhid, nout)).astype(np.float32)
            b_hh = np.zeros((nhid,), dtype=np.float32)
            b_hy = np.zeros((nout,), dtype=np.float32)
            self.act_name = "tanh"
        elif init == "smart_tanh":
            # See Pascanu's 2013 ICML paper for more details on this initialization.
            W_uh = rng.normal(loc=0.0, scale=0.01, size=(nin, nhid)).astype(np.float32)
            W_hh = rng.normal(loc=0.0, scale=0.01, size=(nhid, nhid)).astype(np.float32)
            # sparsify each row: keep first 15 indices of a random permutation (others set to 0)
            for dx in range(nhid):
                spng = rng.permutation(nhid)
                W_hh[dx, spng[15:]] = 0.0
            sr = spectral_radius(W_hh)
            if sr > 0:
                W_hh = (0.95 * W_hh / sr).astype(np.float32)
            W_hy = rng.normal(loc=0.0, scale=0.01, size=(nhid, nout)).astype(np.float32)
            b_hh = np.zeros((nhid,), dtype=np.float32)
            b_hy = np.zeros((nout,), dtype=np.float32)
            self.act_name = "tanh"
        else:
            raise ValueError(f"Unknown init={init}. Choose from sigmoid, test, basic_tanh, smart_tanh")

        # Parameters with same naming as Theano code
        self.W_uh = nn.Parameter(torch.tensor(W_uh, dtype=dtype, device=device))
        self.W_hh = nn.Parameter(torch.tensor(W_hh, dtype=dtype, device=device))
        self.W_hy = nn.Parameter(torch.tensor(W_hy, dtype=dtype, device=device))
        self.b_hh = nn.Parameter(torch.tensor(b_hh, dtype=dtype, device=device))
        self.b_hy = nn.Parameter(torch.tensor(b_hy, dtype=dtype, device=device))

    def act(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_name == "sigmoid":
            return torch.sigmoid(x)
        if self.act_name == "tanh":
            return torch.tanh(x)
        if self.act_name == "identity":
            return x
        raise RuntimeError("bad act_name")

    def act_deriv_from_h(self, h: torch.Tensor) -> torch.Tensor:
        """
        Derivative of activation wrt pre-activation, expressed using h = act(preact).
        Matches Theano code:
          sigmoid: h * (1 - h)
          tanh: 1 - h^2
          identity: 1
        """
        if self.act_name == "sigmoid":
            return h * (1.0 - h)
        if self.act_name == "tanh":
            return 1.0 - h * h
        if self.act_name == "identity":
            return torch.ones_like(h)
        raise RuntimeError("bad act_name")

    # DO THIS
    def forward(self, u: torch.Tensor):
        T, B, _ = u.shape
        h_t = torch.zeros(B, self.nhid, device=u.device, dtype=u.dtype)
        h_seq = []

        for t in range(T):
            # Recurrence: h_t = act(x_t*W_uh + h_{t-1}*W_hh + b_hh) 
            pre_act = torch.mm(u[t], self.W_uh) + torch.mm(h_t, self.W_hh) + self.b_hh
            h_t = self.act(pre_act)
            h_seq.append(h_t)

        h_seq = torch.stack(h_seq) # (T, B, H)
        
        # Output head: o_t = h_t*W_hy + b_hy [cite: 20]
        if self.classif_type == "softmax":
            logits = torch.mm(h_seq.view(T * B, self.nhid), self.W_hy) + self.b_hy
        else:
            logits = torch.mm(h_seq[-1], self.W_hy) + self.b_hy
            
        return logits, h_seq
    supports_omega: bool = True

    def saturation_distance_from_h(self, h: torch.Tensor) -> torch.Tensor:
        """Elementwise distance-to-saturation for the nonlinearity.

        For tanh: min(1-h, 1+h). For sigmoid: min(h, 1-h).
        Smaller => more saturated.
        """
        if self.act_name == "sigmoid":
            return _sigmoid_saturation_distance(h)
        # tanh and identity: for identity we still report tanh-style distance,
        # but it'll typically be >1; callers may ignore.
        return _tanh_saturation_distance(h)

    # DO THIS
    def recurrent_weight_for_rho(self) -> torch.Tensor:
        # This needs to return the recurrent weight matrix.
         return self.W_hh [cite: 22]

    def numpy_state(self) -> dict:
        return {
            "W_hh": self.W_hh.detach().cpu().numpy(),
            "W_uh": self.W_uh.detach().cpu().numpy(),
            "W_hy": self.W_hy.detach().cpu().numpy(),
            "b_hh": self.b_hh.detach().cpu().numpy(),
            "b_hy": self.b_hy.detach().cpu().numpy(),
            "act_name": np.array(self.act_name),
            "classif_type": np.array(self.classif_type),
            "model_type": np.array("rnn"),
        }


class GRUModel(nn.Module):
    """A minimal GRU implemented in the same (T,B,*) style as VanillaRNN.

    This is meant for *behavioral comparison* in the assignment (vanishing/
    exploding, saturation), not as a faithful port of the paper's Omega
    regularizer. We therefore mark supports_omega=False.
    """

    supports_omega: bool = False

    def __init__(
        self,
        nin: int,
        nout: int,
        nhid: int,
        init: str = "smart_tanh",
        classif_type: str = "lastSoftmax",
        rng: np.random.RandomState | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.nin = int(nin)
        self.nout = int(nout)
        self.nhid = int(nhid)
        self.init = init
        self.classif_type = classif_type

        if rng is None:
            rng = np.random.RandomState(1234)

        # Gates: update z, reset r
        W_uz = rng.normal(0, 0.01, size=(nin, nhid))
        W_hz = rng.normal(0, 0.01 / 2, size=(nhid, nhid))
        b_z = np.zeros((nhid,), dtype=np.float64) - 2

        W_ur = rng.normal(0, 0.01, size=(nin, nhid))
        W_hr = rng.normal(0, 0.01 / 2, size=(nhid, nhid))
        b_r = np.zeros((nhid,), dtype=np.float64)

        # Candidate
        W_hh = rng.normal(0, 0.01, size=(nhid, nhid))
        W_uh = rng.normal(0, 0.01, size=(nin, nhid))
        b_h = np.zeros((nhid,), dtype=np.float64)

        W_hy = rng.normal(0, 0.01, size=(nhid, nout))
        b_y = np.zeros((nout,), dtype=np.float64)

        # Torch params
        self.W_uz = nn.Parameter(torch.tensor(W_uz, dtype=dtype, device=device))
        self.W_hz = nn.Parameter(torch.tensor(W_hz, dtype=dtype, device=device))
        self.b_z = nn.Parameter(torch.tensor(b_z, dtype=dtype, device=device))

        self.W_ur = nn.Parameter(torch.tensor(W_ur, dtype=dtype, device=device))
        self.W_hr = nn.Parameter(torch.tensor(W_hr, dtype=dtype, device=device))
        self.b_r = nn.Parameter(torch.tensor(b_r, dtype=dtype, device=device))

        self.W_uh = nn.Parameter(torch.tensor(W_uh, dtype=dtype, device=device))
        self.W_hh = nn.Parameter(torch.tensor(W_hh, dtype=dtype, device=device))
        self.b_h = nn.Parameter(torch.tensor(b_h, dtype=dtype, device=device))

        self.W_hy = nn.Parameter(torch.tensor(W_hy, dtype=dtype, device=device))
        self.b_y = nn.Parameter(torch.tensor(b_y, dtype=dtype, device=device))

    def saturation_distance_from_h(self, h: torch.Tensor) -> torch.Tensor:
        # hidden values remain in [-1,1]
        return _tanh_saturation_distance(h)

    # DO THIS
    def recurrent_weight_for_rho(self) -> torch.Tensor:
        # This needs to return the recurrent weight matrix. There are multiple in
        # the case of the GRU: return the candidate one similar to the RNN case.
        return self.W_hh [cite: 34]

    def numpy_state(self) -> dict:
        return {
            "W_uz": self.W_uz.detach().cpu().numpy(),
            "W_hz": self.W_hz.detach().cpu().numpy(),
            "b_z": self.b_z.detach().cpu().numpy(),
            "W_ur": self.W_ur.detach().cpu().numpy(),
            "W_hr": self.W_hr.detach().cpu().numpy(),
            "b_r": self.b_r.detach().cpu().numpy(),
            "W_uh": self.W_uh.detach().cpu().numpy(),
            "W_hh": self.W_hh.detach().cpu().numpy(),
            "b_h": self.b_h.detach().cpu().numpy(),
            "W_hy": self.W_hy.detach().cpu().numpy(),
            "b_y": self.b_y.detach().cpu().numpy(),
            "init": np.array(self.init),
            "classif_type": np.array(self.classif_type),
            "model_type": np.array("gru"),
        }

    # DO THIS
    def forward(self, u: torch.Tensor, return_extras: bool = False):
        T, B, _ = u.shape
        h_t = torch.zeros(B, self.nhid, device=u.device, dtype=u.dtype)
        h_seq, z_seq, r_seq, h_tilde_pre_seq = [], [], [], []

        for t in range(T):
            x_t = u[t]
            # Equations from assignment 
            z_pre = torch.mm(x_t, self.W_uz) + torch.mm(h_t, self.W_hz) + self.b_z
            r_pre = torch.mm(x_t, self.W_ur) + torch.mm(h_t, self.W_hr) + self.b_r
            
            z_t = torch.sigmoid(z_pre)
            r_t = torch.sigmoid(r_pre)
            
            h_tilde_pre = torch.mm(x_t, self.W_uh) + torch.mm(r_t * h_t, self.W_hh) + self.b_h
            h_tilde = torch.tanh(h_tilde_pre)
            
            h_t = (1 - z_t) * h_t + z_t * h_tilde
            
            h_seq.append(h_t)
            if return_extras:
                z_seq.append(z_pre)
                r_seq.append(r_pre)
                h_tilde_pre_seq.append(h_tilde_pre)

        h_seq = torch.stack(h_seq)
        
        if self.classif_type == "softmax":
            logits = torch.mm(h_seq.view(T * B, self.nhid), self.W_hy) + self.b_y
        else:
            logits = torch.mm(h_seq[-1], self.W_hy) + self.b_y

        if return_extras:
            extras = {
                "z": torch.stack(z_seq),
                "r": torch.stack(r_seq),
                "h_tilde": torch.stack(h_tilde_pre_seq)
            }
            return logits, h_seq, extras
        
        return logits, h_seq


def make_model(
    model_type: str,
    nin: int,
    nout: int,
    nhid: int,
    init: str,
    classif_type: str,
    rng: np.random.RandomState,
    dtype: torch.dtype,
    device: torch.device,
):
    model_type = model_type.lower()
    if model_type in {"rnn", "vanilla", "vanillarnn"}:
        return VanillaRNN(
            nin=nin,
            nout=nout,
            nhid=nhid,
            init=init,
            classif_type=classif_type,
            rng=rng,
            dtype=dtype,
            device=device,
        )
    if model_type in {"gru"}:
        return GRUModel(
            nin=nin,
            nout=nout,
            nhid=nhid,
            init=init,
            classif_type=classif_type,
            rng=rng,
            dtype=dtype,
            device=device,
        )
    raise ValueError(f"Unknown model_type={model_type}")
