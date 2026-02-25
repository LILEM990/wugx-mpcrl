# LSTM-based dynamics model for summer rabbit hutch control.
# Since no physical model exists for the rabbit hutch, an LSTM neural network
# is trained to learn the input-output dynamics from data.

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class RabbitHutchLSTM(nn.Module):
    """LSTM-based dynamics model for summer rabbit hutch control.

    Since no physical model is available for the rabbit hutch, this LSTM
    learns the discrete-time dynamics from collected transition data.
    The model predicts the residual (delta_x = x_{t+1} - x_t) using a
    recurrent LSTM core followed by a linear output layer.

    State variables (nx = 4):
        x[0]: Indoor temperature (°C)
        x[1]: Indoor relative humidity (%)
        x[2]: Indoor CO2 concentration (ppm)
        x[3]: Indoor NH3 concentration (ppm)

    Control inputs (nu = 3):
        u[0]: Ventilation rate (0 – 1, fraction of max)
        u[1]: Cooling power (0 – 1, fraction of max)
        u[2]: Shade/blind position (0 – 1, fraction of max)

    Environmental disturbances (nd = 3):
        d[0]: Outdoor temperature (°C)
        d[1]: Solar radiation (W/m²)
        d[2]: Outdoor relative humidity (%)
    """

    nx: int = 4
    nu: int = 3
    nd: int = 3

    def __init__(self, hidden_size: int = 16, num_layers: int = 1) -> None:
        """Initialise the LSTM dynamics model.

        Parameters
        ----------
        hidden_size : int
            Number of LSTM hidden units per layer.
        num_layers : int
            Number of stacked LSTM layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        input_size = self.nx + self.nu + self.nd

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, self.nx)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        d: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-step prediction.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, nx)
            Current state.
        u : torch.Tensor, shape (batch, nu)
            Control input applied at the current step.
        d : torch.Tensor, shape (batch, nd)
            Environmental disturbance at the current step.
        hidden : tuple | None
            LSTM hidden state (h, c); initialised to zeros when ``None``.

        Returns
        -------
        tuple
            ``(x_next, hidden)`` where *x_next* has shape ``(batch, nx)``
            and *hidden* is the updated LSTM hidden state.
        """
        inp = torch.cat([x, u, d], dim=-1).unsqueeze(1)  # (batch, 1, input)
        out, hidden = self.lstm(inp, hidden)
        delta_x = self.fc(out.squeeze(1))  # residual prediction
        x_next = x + delta_x
        return x_next, hidden

    # ------------------------------------------------------------------
    # Numpy interface (used inside the gym environment)
    # ------------------------------------------------------------------

    def predict(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        hidden: Optional[tuple] = None,
    ) -> tuple[np.ndarray, tuple]:
        """Single-step prediction via numpy arrays (no gradient).

        Parameters
        ----------
        x : np.ndarray, shape (nx,)
        u : np.ndarray, shape (nu,)
        d : np.ndarray, shape (nd,)
        hidden : tuple | None
            PyTorch LSTM hidden state.

        Returns
        -------
        tuple
            ``(x_next, hidden)`` — next state as a numpy array and the
            updated hidden state.
        """
        with torch.no_grad():
            x_t = torch.FloatTensor(x).unsqueeze(0)
            u_t = torch.FloatTensor(u).unsqueeze(0)
            d_t = torch.FloatTensor(d).unsqueeze(0)
            x_next_t, hidden_out = self.forward(x_t, u_t, d_t, hidden)
        return x_next_t.squeeze(0).numpy(), hidden_out

    # ------------------------------------------------------------------
    # Training helper
    # ------------------------------------------------------------------

    def train_on_transitions(
        self,
        transitions: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> list[float]:
        """Train the LSTM on a list of ``(x, u, d, x_next)`` transitions.

        Parameters
        ----------
        transitions : list of (x, u, d, x_next)
            Each element is one observed transition.
        epochs : int
            Training epochs.
        lr : float
            Adam learning-rate.
        batch_size : int
            Mini-batch size.
        verbose : bool
            Print loss every 20 epochs.

        Returns
        -------
        list[float]
            MSE loss at each epoch.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X = torch.FloatTensor(np.array([t[0] for t in transitions]))
        U = torch.FloatTensor(np.array([t[1] for t in transitions]))
        D = torch.FloatTensor(np.array([t[2] for t in transitions]))
        X_next = torch.FloatTensor(np.array([t[3] for t in transitions]))

        losses: list[float] = []
        n = X.shape[0]
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                x_pred, _ = self.forward(X[idx], U[idx], D[idx])
                loss = criterion(x_pred, X_next[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            losses.append(epoch_loss / n)
            if verbose and epoch % 20 == 0:
                print(f"  epoch {epoch:4d}/{epochs}  loss={losses[-1]:.6f}")
        return losses

    # ------------------------------------------------------------------
    # Weight export (for CasADi MPC)
    # ------------------------------------------------------------------

    def get_casadi_weights(self) -> dict[str, np.ndarray]:
        """Extract all LSTM and output-layer weights as numpy arrays.

        The weight layout follows PyTorch convention: the parameter
        ``weight_ih_l{n}`` stores [W_i; W_f; W_g; W_o] stacked along
        dim 0, each of shape ``(hidden_size, input_size)``.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``W_ih_{l}``, ``W_hh_{l}``, ``b_ih_{l}``, ``b_hh_{l}``
            for each layer ``l``, plus ``W_fc`` and ``b_fc``.
        """
        weights: dict[str, np.ndarray] = {}
        for layer in range(self.num_layers):
            weights[f"W_ih_{layer}"] = (
                self.lstm.__getattr__(f"weight_ih_l{layer}").detach().numpy()
            )
            weights[f"W_hh_{layer}"] = (
                self.lstm.__getattr__(f"weight_hh_l{layer}").detach().numpy()
            )
            weights[f"b_ih_{layer}"] = (
                self.lstm.__getattr__(f"bias_ih_l{layer}").detach().numpy()
            )
            weights[f"b_hh_{layer}"] = (
                self.lstm.__getattr__(f"bias_hh_l{layer}").detach().numpy()
            )
        weights["W_fc"] = self.fc.weight.detach().numpy()
        weights["b_fc"] = self.fc.bias.detach().numpy()
        return weights

    # ------------------------------------------------------------------
    # Physical bounds (for the environment and MPC)
    # ------------------------------------------------------------------

    @staticmethod
    def get_u_min() -> np.ndarray:
        """Minimum control inputs ``[ventilation, cooling, shade]``."""
        return np.zeros(3)

    @staticmethod
    def get_u_max() -> np.ndarray:
        """Maximum control inputs ``[ventilation, cooling, shade]``."""
        return np.ones(3)

    @staticmethod
    def get_du_lim() -> np.ndarray:
        """Maximum allowed per-step change in control inputs."""
        return 0.1 * RabbitHutchLSTM.get_u_max()

    @staticmethod
    def get_x_min() -> np.ndarray:
        """Lower bound on state ``[temp(°C), hum(%), CO2(ppm), NH3(ppm)]``."""
        return np.array([0.0, 0.0, 0.0, 0.0])

    @staticmethod
    def get_x_max() -> np.ndarray:
        """Upper bound on state."""
        return np.array([50.0, 100.0, 5000.0, 100.0])

    @staticmethod
    def get_y_min_summer() -> np.ndarray:
        """Comfort lower bounds for rabbits in summer."""
        return np.array([18.0, 40.0, 400.0, 0.0])

    @staticmethod
    def get_y_max_summer() -> np.ndarray:
        """Comfort upper bounds for rabbits in summer."""
        return np.array([28.0, 70.0, 3000.0, 25.0])

    @staticmethod
    def get_y_range_summer() -> np.ndarray:
        """Range of comfort bounds (used for normalisation)."""
        return RabbitHutchLSTM.get_y_max_summer() - RabbitHutchLSTM.get_y_min_summer()

    # ------------------------------------------------------------------
    # Synthetic data generation (for pre-training without real data)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_synthetic_transitions(
        num_steps: int = 5000,
        np_random: np.random.Generator | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Generate synthetic transition data from a simplified thermal model.

        The simplified physics captures the key thermal and mass-transfer
        dynamics of a summer rabbit hutch:
        - Temperature is driven by solar gain, ventilation cooling, and
          active cooling.
        - Humidity is driven by respiration and ventilation.
        - CO2 accumulates via respiration and is removed by ventilation.
        - NH3 accumulates from waste and is removed by ventilation.

        Parameters
        ----------
        num_steps : int
            Number of transitions to generate.
        np_random : np.random.Generator | None
            Random number generator.  Uses ``np.random.default_rng()`` when
            ``None``.

        Returns
        -------
        list of (x, u, d, x_next) tuples
        """
        rng = np.random.default_rng(np_random)
        ts = 60.0 * 15.0  # 15-minute step (seconds)

        transitions = []
        # start in comfortable conditions
        x = np.array([24.0, 55.0, 800.0, 5.0])
        for step in range(num_steps):
            # random control action
            u = rng.uniform(0.0, 1.0, size=3)
            # summer disturbance: high outdoor temp, noon-ish radiation
            phase = (step % 96) / 96 * 2 * np.pi
            d = np.array(
                [
                    32.0 + 8.0 * np.sin(phase - np.pi / 2),  # outdoor temp
                    max(0.0, 800.0 * np.sin(phase)),  # solar radiation
                    60.0 + 10.0 * np.cos(phase),  # outdoor humidity
                ]
            )

            # simplified thermal physics
            vent = u[0]  # ventilation fraction
            cool = u[1]  # cooling fraction
            shade = u[2]  # shade fraction

            # temperature dynamics (°C/step)
            solar_gain = 0.05 * (1.0 - shade) * d[1] / 800.0
            vent_exchange = 0.3 * vent * (d[0] - x[0])
            cooling_effect = -0.4 * cool
            dT = (solar_gain + vent_exchange + cooling_effect) * ts / 3600.0

            # humidity dynamics (%/step)
            rabbit_transpiration = 0.02  # %/step from 100 rabbits
            vent_hum = 0.25 * vent * (d[2] - x[1])
            dH = (rabbit_transpiration + vent_hum) * ts / 3600.0

            # CO2 dynamics (ppm/step)
            rabbit_co2 = 5.0  # ppm/step generation
            vent_co2 = 0.4 * vent * (400.0 - x[2])
            dCO2 = (rabbit_co2 + vent_co2) * ts / 3600.0

            # NH3 dynamics (ppm/step)
            waste_nh3 = 0.02  # ppm/step generation
            vent_nh3 = 0.4 * vent * (0.0 - x[3])
            dNH3 = (waste_nh3 + vent_nh3) * ts / 3600.0

            x_next = x + np.array([dT, dH, dCO2, dNH3])
            x_next = np.clip(
                x_next,
                RabbitHutchLSTM.get_x_min(),
                RabbitHutchLSTM.get_x_max(),
            )
            transitions.append((x.copy(), u.copy(), d.copy(), x_next.copy()))
            x = x_next

        return transitions
