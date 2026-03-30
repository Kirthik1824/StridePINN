"""
pinn.py — Physics-Informed Neural Network for gait dynamics.

The PINN is a latent-variable model with three jointly trained components:

  1. Encoder E_φ : ℝ⁹ → ℝ² — maps first time-step to 2-D latent IC z(0)
  2. Neural ODE f_θ : ℝ² → ℝ² — Hopf normal form + residual MLP
  3. Decoder D_ψ : ℝ² → ℝ⁹ — linear map from latent state to all accel channels

Loss = L_data + λ_cyc · L_cyc + λ_φ · L_φ + λ_s · L_s + λ_r · L_radius

The Hopf normal form provides an inductive bias toward stable limit cycles,
ensuring the latent dynamics model oscillatory gait patterns.
"""

import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint


# -----------------------------------------------------------------
#  Stage 1 — Encoder E_φ
# -----------------------------------------------------------------
class GaitEncoder(nn.Module):
    """
    MLP encoder: maps a single 9-D IMU observation to a 2-D latent IC.

    Architecture: 9 → hidden → hidden → 2 (ReLU activations)
    """

    def __init__(self, in_dim: int = 9, hidden: int = 128, latent_dim: int = 2):
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
    Hopf Normal Form ODE with learnable residual correction.

    Base dynamics (Hopf normal form):
        ẋ = α(μ - r²)x - ωy
        ẏ = α(μ - r²)y + ωx

    This guarantees a stable limit cycle at radius √μ with angular
    frequency ω. A small residual MLP adds expressivity for non-ideal
    oscillator dynamics.

    Modes:
        - "hopf": Hopf normal form + residual MLP (recommended)
        - "mlp":  Free-form MLP (original, for ablation)
    """

    def __init__(self, latent_dim: int = 2, hidden: int = 64, mode: str = "hopf"):
        super().__init__()
        self.mode = mode
        self.latent_dim = latent_dim

        # Residual MLP (used in both modes)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )

        if mode == "hopf":
            # Learnable Hopf parameters
            self.mu = 1.0  # Fixed target radius² = 1.0
            self.log_alpha = nn.Parameter(torch.tensor(0.0))
            self.log_omega = nn.Parameter(torch.tensor(np.log(2.0 * np.pi * 1.5)))
            self.log_epsilon = nn.Parameter(torch.tensor(-2.0))  # exp(-2)≈0.13, slightly stronger
            
            # Learnable center for the limit cycle tracking
            self.cx = nn.Parameter(torch.tensor(0.0))
            self.cy = nn.Parameter(torch.tensor(0.0))

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "mlp":
            return self.net(z)

        # --- Hopf Normal Form ---
        x, y = z[..., 0], z[..., 1]

        # Constrain parameters
        mu = self.mu
        alpha = torch.exp(self.log_alpha)
        
        if hasattr(self, 'current_omega') and self.current_omega is not None:
            omega = self.current_omega
        else:
            omega = torch.exp(self.log_omega)
            
        cx, cy = self.cx, self.cy

        # Center trajectories
        x_c, y_c = x - cx, y - cy
        r2 = x_c ** 2 + y_c ** 2 + 1e-8

        # Hopf dynamics relative to center
        radial = alpha * (mu - r2)
        dx = radial * x_c - omega * y_c
        dy = radial * y_c + omega * x_c

        # CRITICAL FIX: The residual MLP (self.net) previously acted as a "physics assassin",
        # learning to perfectly cancel the [dx, dy] rotation so it could minimize L_traj as a static dot.
        # By exclusively using dz_hopf, we mathematically enforce the limit-cycle topology.
        dz = torch.stack([dx, dy], dim=-1)

        # Clamp to prevent blow-up during integration
        return torch.clamp(dz, -20.0, 20.0)


# -----------------------------------------------------------------
#  Stage 3 — Decoder D_ψ
# -----------------------------------------------------------------
class GaitDecoder(nn.Module):
    """Linear decoder from 2D latent to ALL 9 accelerometer channels."""
    def __init__(self, latent_dim: int = 2, out_dim: int = 9):
        super().__init__()
        self.linear = nn.Linear(latent_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


# -----------------------------------------------------------------
#  Full PINN
# -----------------------------------------------------------------
class GaitPINN(nn.Module):
    def __init__(
        self,
        in_dim: int = 9,
        latent_dim: int = 2,
        encoder_hidden: int = 128,
        ode_hidden: int = 64,
        decoder_out: int = 9,
        ode_method: str = "rk4",
        ode_rtol: float = 1e-4,
        ode_atol: float = 1e-5,
        ode_step_size: float = 0.1,
        ode_mode: str = "hopf",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.ode_method = ode_method
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol
        self.ode_step_size = ode_step_size

        self.encoder = GaitEncoder(in_dim, encoder_hidden, latent_dim)
        self.ode_func = NeuralODEFunc(latent_dim, ode_hidden, mode=ode_mode)
        self.decoder = GaitDecoder(latent_dim, decoder_out)
        
        # Frequency head (predicts cadence drift)
        self.omega_head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor = None,
    ) -> dict:
        B, T, C = x.shape
        
        # 1. Encode every timestep independently (temporal convolution / time-distributed dense)
        # Flatten time and batch to process through standard Linear layers
        x_flat = x.reshape(B * T, C)
        z_data_flat = self.encoder(x_flat)
        
        # Reshape and permute back to (T, B, latent_dim) for downstream compatibility
        z_data = z_data_flat.reshape(B, T, self.latent_dim).permute(1, 0, 2)
        
        # 2. Predict adaptive frequency (omega) from the window summary
        # We use global average pooling of latent features to estimate the window frequency
        z_mean = z_data.mean(dim=0) # (B, latent_dim)
        omega_raw = self.omega_head(z_mean).squeeze(-1) # (B,)
        
        # Constrain omega to 0.5Hz - 3.0Hz (2.0*pi * freq)
        # Using a sigmoid-like scaling to center it around 1.5Hz
        omega = (torch.sigmoid(omega_raw) * 2.5 + 0.5) * (2.0 * np.pi)
        
        # 3. We NO LONGER do a full 3.0s rigid ODE rollout here.
        # Rigid rollouts desynchronize with biological phase micro-jitter,
        # forcing the encoder to collapse to a static dot.
        # Instead, we will evaluate the local Vector Field matching in compute_loss!
        
        # 4. Decode every timestep
        x_hat_flat = self.decoder(z_data_flat)
        x_hat = x_hat_flat.reshape(B, T, -1).permute(1, 0, 2)
        
        return {
            "z_encoded": z_data, 
            "x_hat": x_hat,
            "omega": omega,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        lambda_cyc: float = 1.0,
        lambda_phi: float = 10.0,
        lambda_smooth: float = 0.1,
        lambda_radius: float = 1.0,
    ) -> dict:
        out = self.forward(x)
        z_encoded = out["z_encoded"]   # (T, B, 2)
        x_hat = out["x_hat"]           # (T, B, out_dim)

        x_hat_bt = x_hat.permute(1, 0, 2)
        L_data = torch.mean((target - x_hat_bt) ** 2)

        # L_phi — phase monotonicity (using first 2 dims as phase plane)
        z_bt = z_encoded.permute(1, 0, 2)
        if hasattr(self.ode_func, 'cx'):
            cx, cy = self.ode_func.cx, self.ode_func.cy
            phase = torch.atan2(z_bt[:, :, 1] - cy, z_bt[:, :, 0] - cx)
        else:
            phase = torch.atan2(z_bt[:, :, 1], z_bt[:, :, 0])
            
        d_phase = phase[:, 1:] - phase[:, :-1]
        d_phase = torch.atan2(torch.sin(d_phase), torch.cos(d_phase))
        L_phi = torch.mean(torch.clamp(-d_phase, min=0) ** 2)

        # L_s — smoothness
        accel = z_bt[:, 2:] - 2 * z_bt[:, 1:-1] + z_bt[:, :-2]
        L_smooth = torch.mean(torch.sum(accel ** 2, dim=-1))

        # L_radius — force trajectories to live near target radius
        target_r2 = 1.0
        if hasattr(self.ode_func, 'cx'):
            cx, cy = self.ode_func.cx, self.ode_func.cy
            r2 = (z_encoded[..., 0] - cx)**2 + (z_encoded[..., 1] - cy)**2
        else:
            r2 = torch.sum(z_encoded ** 2, dim=-1)  # (T, B)
        L_radius = torch.mean((r2 - target_r2) ** 2)

        # L_ode -- Latent Vector Field Alignment
        # Instead of a global 3.0s rigid rollout (which breaks against biological jitter),
        # we evaluate the discrete time derivative of the encoder and match it locally
        # to the mathematical vector field.
        dt = 3.0 / z_encoded.shape[0] # Time delta per frame
        dz_data = (z_encoded[1:, :, :] - z_encoded[:-1, :, :]) / dt
        
        # We need the vector field at exactly the encoded points
        # t is a dummy variable since the ODE is autonomous
        self.ode_func.current_omega = out["omega"]
        
        # Evaluate vector field over entire sequence directly (shapes rely on broadcasting)
        z_eval = z_encoded[:-1, :, :]  # (T-1, B, 2)
        dz_hopf = self.ode_func(None, z_eval)  # Returns (T-1, B, 2)
        
        self.ode_func.current_omega = None
        
        L_ode = torch.mean((dz_data - dz_hopf) ** 2)

        # Guide Item: Latent Covariance (Isotropy) Loss -- forcefully ensures circular spread
        z_flat = z_bt.reshape(-1, self.latent_dim)
        z_flat_centered = z_flat - z_flat.mean(dim=0)
        cov = (z_flat_centered.T @ z_flat_centered) / (z_flat.shape[0] - 1)
        I = torch.eye(self.latent_dim, device=z_flat.device)
        L_iso = torch.mean((cov - 0.5 * I) ** 2)

        L_total = (L_data
                   + lambda_phi * L_phi
                   + lambda_smooth * L_smooth
                   + lambda_radius * L_radius
                   + 1.0 * L_ode  # Substituted traj rollout for local derivative match
                   + 1.0 * L_iso) # Geometric decorrelation keeps it a perfect circle

        return {
            "total": L_total,
            "data": L_data,
            "phi": L_phi,
            "smooth": L_smooth,
            "radius": L_radius,
            "ode": L_ode,
            "iso": L_iso,
        }

    def compute_anomaly_scores(self, x: torch.Tensor) -> dict:
        with torch.no_grad():
            out = self.forward(x)
            z_encoded = out["z_encoded"]
            T_len, B, D = z_encoded.shape

            z_bt = z_encoded.permute(1, 0, 2)

            # True Structural Residual: Mathematical divergence between data and Physics prior
            # Evaluated locally via Vector Field alignment (handles biological jitter)
            dt = 3.0 / T_len
            dz_data = (z_bt[:, 1:, :] - z_bt[:, :-1, :]) / dt
            
            # Predict the ideal vector field at current encoded points
            self.ode_func.current_omega = out["omega"]
            
            # Evaluate on (T, B, D) shape to cleanly broadcast omega (B,)
            z_eval = z_encoded[:-1, :, :]
            dz_hopf_tb = self.ode_func(None, z_eval)
            self.ode_func.current_omega = None
            
            dz_hopf_bt = dz_hopf_tb.permute(1, 0, 2) # Back to (B, T-1, D)
            
            # Error magnitude per timestep (padded with 0 at start to maintain seq length)
            raw_residual = torch.norm(dz_data - dz_hopf_bt, dim=-1)
            residual = torch.cat([torch.zeros(B, 1, device=x.device), raw_residual], dim=1)
            residual_mean = residual.mean(dim=-1)
            
            # Phase: departure from expected angular advance
            if hasattr(self.ode_func, 'cx'):
                cx, cy = self.ode_func.cx, self.ode_func.cy
                phase = torch.atan2(z_bt[:, :, 1] - cy, z_bt[:, :, 0] - cx)
                r2 = (z_bt[..., 0] - cx)**2 + (z_bt[..., 1] - cy)**2
            else:
                phase = torch.atan2(z_bt[:, :, 1], z_bt[:, :, 0])
                r2 = z_bt[..., 0]**2 + z_bt[..., 1]**2
                
            d_phase = phase[:, 1:] - phase[:, :-1]
            d_phase = torch.atan2(torch.sin(d_phase), torch.cos(d_phase))
            
            # Expected d_phase is roughly omega * dt
            # We measure the variance/uncertainty in phase advance
            phase_var = torch.std(d_phase, dim=-1)
            
            # Radius features
            r = torch.sqrt(r2 + 1e-8)
            var_r = torch.var(r, dim=-1)
            mean_abs_r_1 = torch.mean(torch.abs(r - 1.0), dim=-1)
            mean_r2 = torch.mean(r2, dim=-1)

            return {
                "residual_max": residual_mean,
                "phase_advance": phase_var,
                "var_r": var_r,
                "mean_abs_r_1": mean_abs_r_1,
                "mean_r2": mean_r2,
                "z_traj": z_encoded,
            }
