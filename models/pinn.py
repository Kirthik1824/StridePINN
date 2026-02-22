"""
pinn.py — Physics-Informed Neural Network for gait dynamics.

The PINN is a latent-variable model with three jointly trained components:

  1. Encoder E_φ : ℝ⁹ → ℝ² — maps first time-step to 2-D latent IC z(0)
  2. Neural ODE f_θ : ℝ² → ℝ² — autonomous vector field (tanh MLP)
  3. Decoder D_ψ : ℝ² → ℝ³ — linear map from latent state to ankle accel

Loss = L_data + λ_cyc · L_cyc + λ_φ · L_φ + λ_s · L_s

Total parameters: ~5k (30× fewer than CNN-LSTM)
"""

import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint_adjoint as odeint


# -----------------------------------------------------------------
#  Stage 1 — Encoder E_φ
# -----------------------------------------------------------------
class GaitEncoder(nn.Module):
    """
    MLP encoder: maps a single 9-D IMU observation to a 2-D latent IC.

    Architecture: 9 → 64 → 64 → 2 (ReLU activations)
    """

    def __init__(self, in_dim: int = 9, hidden: int = 64, latent_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 9) — IMU observation at t=0

        Returns:
            z0: (B, 2) — latent initial condition
        """
        return self.net(x)


# -----------------------------------------------------------------
#  Stage 2 — Neural ODE Vector Field f_θ
# -----------------------------------------------------------------
class NeuralODEFunc(nn.Module):
    """
    Smooth autonomous vector field parameterised as a tanh-MLP.

    Architecture: 2 → 32 → 2 (tanh activation)

    tanh is preferred over ReLU because it produces a smooth, bounded
    vector field — necessary for generating smooth curved orbits
    (the limit cycle) in the 2-D latent plane.
    """

    def __init__(self, latent_dim: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: scalar time (required by torchdiffeq, unused for autonomous ODE)
            z: (B, 2) — current latent state

        Returns:
            dz_dt: (B, 2) — time derivative of latent state
        """
        return self.net(z)


# -----------------------------------------------------------------
#  Stage 3 — Decoder D_ψ
# -----------------------------------------------------------------
class GaitDecoder(nn.Module):
    """
    Linear decoder: maps latent state → 3-axis ankle acceleration.

    A linear decoder strengthens interpretability: if reconstruction
    is accurate, the 2-D latent space must itself geometrically encode
    the gait oscillation.
    """

    def __init__(self, latent_dim: int = 2, out_dim: int = 3):
        super().__init__()
        self.linear = nn.Linear(latent_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (..., 2) — latent state(s)

        Returns:
            x_hat: (..., 3) — reconstructed ankle accel
        """
        return self.linear(z)


# -----------------------------------------------------------------
#  Full PINN
# -----------------------------------------------------------------
class GaitPINN(nn.Module):
    """
    Physics-Informed gait model = Encoder + Neural ODE + Decoder.

    Forward pass:
      1. Encode first time-step → z(0)
      2. Integrate ODE over [0, T] → z(t₀), z(t₁), …, z(t_{N-1})
      3. Decode each z(tₖ) → x̂(tₖ)

    Provides compute_loss() returning the full 4-term physics loss.
    """

    def __init__(
        self,
        in_dim: int = 9,
        latent_dim: int = 2,
        encoder_hidden: int = 64,
        ode_hidden: int = 32,
        decoder_out: int = 3,
        ode_method: str = "dopri5",
        ode_rtol: float = 1e-4,
        ode_atol: float = 1e-5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.ode_method = ode_method
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol

        self.encoder = GaitEncoder(in_dim, encoder_hidden, latent_dim)
        self.ode_func = NeuralODEFunc(latent_dim, ode_hidden)
        self.decoder = GaitDecoder(latent_dim, decoder_out)

    def forward(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor = None,
    ) -> dict:
        """
        Full forward pass.

        Args:
            x: (B, T, 9) — full window of 9-channel IMU data
            t_span: (T,) — query time-points; defaults to [0, 1, ..., T-1]

        Returns:
            dict with keys:
              z0:      (B, 2)    — latent initial condition
              z_traj:  (T, B, 2) — latent trajectory
              x_hat:   (T, B, 3) — reconstructed ankle accel
        """
        B, T, C = x.shape

        # Encode the first time-step
        z0 = self.encoder(x[:, 0, :])   # (B, 9) → (B, 2)

        # Define integration times
        if t_span is None:
            # Normalise to [0, 1] for numerical stability of ODE
            t_span = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype)

        # Integrate the Neural ODE
        z_traj = odeint(
            self.ode_func,
            z0,
            t_span,
            method=self.ode_method,
            rtol=self.ode_rtol,
            atol=self.ode_atol,
        )  # (T, B, 2)

        # Decode each latent state → ankle accel
        x_hat = self.decoder(z_traj)   # (T, B, 3)

        return {
            "z0": z0,
            "z_traj": z_traj,
            "x_hat": x_hat,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        ankle_target: torch.Tensor,
        lambda_cyc: float = 1.0,
        lambda_phi: float = 10.0,
        lambda_smooth: float = 0.1,
    ) -> dict:
        """
        Compute the full 4-term physics-informed loss.

        Args:
            x:             (B, T, 9) — full window
            ankle_target:  (B, T, 3) — ground-truth ankle acceleration
            lambda_cyc:    weight for periodicity loss
            lambda_phi:    weight for phase monotonicity loss
            lambda_smooth: weight for smoothness regulariser

        Returns:
            dict with keys: total, data, cyc, phi, smooth (all scalar tensors)
        """
        out = self.forward(x)
        z_traj = out["z_traj"]   # (T, B, 2)
        x_hat = out["x_hat"]     # (T, B, 3)

        T, B, _ = z_traj.shape

        # ------------------------------------------------------
        # L_data — MSE reconstruction of ankle acceleration
        # ------------------------------------------------------
        # ankle_target is (B, T, 3); x_hat is (T, B, 3) → permute
        x_hat_bt = x_hat.permute(1, 0, 2)  # (B, T, 3)
        L_data = torch.mean((ankle_target - x_hat_bt) ** 2)

        # ------------------------------------------------------
        # L_cyc — limit-cycle periodicity: ||z(T) - z(0)||²
        # ------------------------------------------------------
        z_start = z_traj[0]    # (B, 2)
        z_end = z_traj[-1]     # (B, 2)
        L_cyc = torch.mean(torch.sum((z_end - z_start) ** 2, dim=-1))

        # ------------------------------------------------------
        # L_phi — phase monotonicity (penalise backward rotation)
        # ------------------------------------------------------
        # Compute phase angle φ(tₖ) = atan2(z₂, z₁)
        z_bt = z_traj.permute(1, 0, 2)  # (B, T, 2)
        phase = torch.atan2(z_bt[:, :, 1], z_bt[:, :, 0])  # (B, T)

        # Unwrap phase differences to handle ±π discontinuities
        d_phase = phase[:, 1:] - phase[:, :-1]  # (B, T-1)
        # Wrap to [-π, π]
        d_phase = torch.atan2(torch.sin(d_phase), torch.cos(d_phase))

        # Penalise negative phase increments (backward rotation)
        L_phi = torch.mean(torch.clamp(-d_phase, min=0) ** 2)

        # ------------------------------------------------------
        # L_s — smoothness: penalise latent acceleration ||z̈||²
        # Second-order central finite differences
        # ------------------------------------------------------
        z_bt2 = z_traj.permute(1, 0, 2)  # (B, T, 2)
        accel = z_bt2[:, 2:] - 2 * z_bt2[:, 1:-1] + z_bt2[:, :-2]  # (B, T-2, 2)
        L_smooth = torch.mean(torch.sum(accel ** 2, dim=-1))

        # ------------------------------------------------------
        # Total loss
        # ------------------------------------------------------
        L_total = (
            L_data
            + lambda_cyc * L_cyc
            + lambda_phi * L_phi
            + lambda_smooth * L_smooth
        )

        return {
            "total": L_total,
            "data": L_data,
            "cyc": L_cyc,
            "phi": L_phi,
            "smooth": L_smooth,
        }

    def compute_anomaly_scores(self, x: torch.Tensor) -> dict:
        """
        Compute FoG detection signals for test windows.

        Two complementary signals:
          1. Dynamics residual r(t) = ||ż_FD(t) - f_θ(z(t))||
          2. Total phase advance ΔΦ

        Args:
            x: (B, T, 9) — test windows

        Returns:
            dict with keys:
              residual_max: (B,) — max dynamics residual per window
              residual_all: (B, T-2) — full residual time-trace
              phase_advance: (B,) — total unwrapped phase advance
              phase_all: (B, T) — raw phase angle time-trace
              z_traj: (T, B, 2) — latent trajectory for plotting
        """
        with torch.no_grad():
            out = self.forward(x)
            z_traj = out["z_traj"]    # (T, B, 2)
            T, B, D = z_traj.shape

            z_bt = z_traj.permute(1, 0, 2)  # (B, T, 2)

            # Dynamics residual via central finite differences
            # ż_FD(tₖ) = (z(tₖ₊₁) - z(tₖ₋₁)) / (2Δt)
            # For normalised time t ∈ [0,1], Δt = 1/(T-1)
            dt = 1.0 / (T - 1)
            z_dot_fd = (z_bt[:, 2:] - z_bt[:, :-2]) / (2 * dt)  # (B, T-2, 2)

            # f_θ(z(tₖ)) at interior points
            z_interior = z_bt[:, 1:-1]  # (B, T-2, 2)
            # Dummy time for autonomous ODE
            t_dummy = torch.zeros(1, device=x.device)
            f_pred = self.ode_func(t_dummy, z_interior.reshape(-1, D))  # (B*(T-2), 2)
            f_pred = f_pred.reshape(B, T - 2, D)

            residual = torch.norm(z_dot_fd - f_pred, dim=-1)  # (B, T-2)
            residual_max = residual.max(dim=-1).values         # (B,)

            # Phase angle
            phase = torch.atan2(z_bt[:, :, 1], z_bt[:, :, 0])  # (B, T)

            # Unwrapped phase advance
            d_phase = phase[:, 1:] - phase[:, :-1]
            d_phase = torch.atan2(torch.sin(d_phase), torch.cos(d_phase))
            phase_advance = d_phase.sum(dim=-1)  # (B,)

            return {
                "residual_max": residual_max,
                "residual_all": residual,
                "phase_advance": phase_advance,
                "phase_all": phase,
                "z_traj": z_traj,
            }
