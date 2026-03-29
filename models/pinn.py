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
        x, y = z[:, 0], z[:, 1]

        # Constrain parameters
        mu = self.mu
        alpha = torch.exp(self.log_alpha)
        
        if hasattr(self, 'current_omega') and self.current_omega is not None:
            omega = self.current_omega
        else:
            omega = torch.exp(self.log_omega)
            
        epsilon = torch.exp(self.log_epsilon)
        cx, cy = self.cx, self.cy

        # Center trajectories
        x_c, y_c = x - cx, y - cy
        r2 = x_c ** 2 + y_c ** 2 + 1e-8

        # Hopf dynamics relative to center
        radial = alpha * (mu - r2)
        dx = radial * x_c - omega * y_c
        dy = radial * y_c + omega * x_c

        dz_hopf = torch.stack([dx, dy], dim=-1)

        # Add small learned residual for expressivity
        dz_residual = epsilon * self.net(z)

        dz = dz_hopf + dz_residual

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
        
        # 3. Guide Fix: True Neural ODE Rollout
        # Integrate forward from the encoded initial state z0
        if t_span is None:
            # A 128-window at 42.6Hz covers exactly 3.0 seconds
            # This ensures omega = 2*pi*f matches physical reality in Hz
            t_span = torch.linspace(0, 3.0, T, device=x.device, dtype=x.dtype)
            
        z0 = z_data[0] # (B, latent_dim)
        
        # Inject the batch-specific omega into the ODE function state for the integration pass
        self.ode_func.current_omega = omega
        
        # We DO NOT set 'step_size' in options.
        # By omitting it, torchdiffeq's rk4 uses the dense 128-point t_span grid inherently,
        # preventing "polygon limit cycles" seen previously with step_size=0.1
        options = {}
            
        z_traj_ode = odeint(
            self.ode_func,
            z0,
            t_span,
            method=self.ode_method,
            rtol=self.ode_rtol,
            atol=self.ode_atol,
            options=options,
        )
        
        # Clear omega to prevent memory leaks or incorrect routing
        self.ode_func.current_omega = None
        
        # 4. Decode every timestep
        x_hat_flat = self.decoder(z_data_flat)
        x_hat = x_hat_flat.reshape(B, T, -1).permute(1, 0, 2)
        
        # z_encoded: the data's path (traced out by the encoder frame-by-frame)
        # z_traj_ode: the mathematically perfect limit cycle rollout
        return {
            "z0": z0,
            "z_encoded": z_data, 
            "z_traj_ode": z_traj_ode,
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
        z_encoded = out["z_encoded"]   # (T, B, D)
        z_traj_ode = out["z_traj_ode"] # (T, B, D)
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

        # L_traj — True Neural ODE Rollout Match (Guide Item 6)
        # Forces the frame-by-frame data representation to align strictly with the limit cycle ODE rollout
        L_traj = torch.mean((z_traj_ode - z_encoded) ** 2)
        
        # Guide Item: Latent Covariance (Isotropy) Loss -- optional but reinforces circular spread
        B_size = z_bt.shape[0]
        z_flat = z_bt.reshape(-1, self.latent_dim)
        z_flat_centered = z_flat - z_flat.mean(dim=0)
        cov = (z_flat_centered.T @ z_flat_centered) / (z_flat.shape[0] - 1)
        I = torch.eye(self.latent_dim, device=z_flat.device)
        # Enforce that the latent space variance is spread as a unit circle 
        # (the determinant of a circle is identity scaled by radius^2. Since radius ~ 1, cov ~ 0.5*I)
        L_iso = torch.mean((cov - 0.5 * I) ** 2)

        L_total = (L_data
                   + lambda_phi * L_phi
                   + lambda_smooth * L_smooth
                   + lambda_radius * L_radius
                   + 10.0 * L_traj  # Rollout structural alignment
                   + 1.0 * L_iso)   # Geometric decorrelation

        return {
            "total": L_total,
            "data": L_data,
            "phi": L_phi,
            "smooth": L_smooth,
            "radius": L_radius,
            "ode": L_traj, # Renamed internally but keeping dict key 'ode' to avoid breakages in pinn_trainer
        }

    def compute_anomaly_scores(self, x: torch.Tensor) -> dict:
        with torch.no_grad():
            out = self.forward(x)
            z_encoded = out["z_encoded"]
            z_traj_ode = out["z_traj_ode"]
            T_len, B, D = z_encoded.shape

            z_bt = z_encoded.permute(1, 0, 2)
            z_bt_ode = z_traj_ode.permute(1, 0, 2)

            # True Structural Residual: Mathematical divergence between data and Physics prior
            # Uses the dynamic ODE rollout instead of localized derivative matching
            residual = torch.norm(z_bt_ode - z_bt, dim=-1)
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
