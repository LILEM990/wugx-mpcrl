# Summer rabbit hutch gym environment.
# The rabbit hutch has no physical model; an LSTM neural network provides
# the discrete-time dynamics for simulation and control.

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium import Env
from gymnasium.spaces import Box

from rabbit_hutch.lstm_model import RabbitHutchLSTM


class RabbitHutchEnv(Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """Summer rabbit hutch environment driven by an LSTM dynamics model.

    The environment simulates the indoor climate of a rabbit hutch during the
    summer season.  Because no first-principles physical model is available,
    an :class:`~rabbit_hutch.lstm_model.RabbitHutchLSTM` is used for the
    discrete-time state transition.

    State variables (nx = 4)
    -------------------------
    x[0] : indoor temperature (°C)
    x[1] : indoor relative humidity (%)
    x[2] : indoor CO₂ concentration (ppm)
    x[3] : indoor NH₃ concentration (ppm)

    Control inputs (nu = 3)
    -------------------------
    u[0] : ventilation rate       (0 – 1 fraction)
    u[1] : cooling power          (0 – 1 fraction)
    u[2] : shade/blind position   (0 – 1 fraction)

    Disturbances (nd = 3)
    -------------------------
    d[0] : outdoor temperature (°C)
    d[1] : solar radiation (W/m²)
    d[2] : outdoor relative humidity (%)

    Summer comfort bounds for rabbits
    -----------------------------------
    Temperature : 18 – 28 °C
    Humidity    : 40 – 70 %
    CO₂         : 400 – 3 000 ppm
    NH₃         : 0 – 25 ppm
    """

    nx: int = RabbitHutchLSTM.nx
    nu: int = RabbitHutchLSTM.nu
    nd: int = RabbitHutchLSTM.nd
    ts: float = 60.0 * 15.0  # 15-minute time step (seconds)
    steps_per_day: int = 24 * 4  # 96 steps / day

    def __init__(
        self,
        num_days: int,
        lstm_model: RabbitHutchLSTM | None = None,
        cost_parameters_dict: dict = {},
        noisy_disturbance: bool = False,
        pretrain_lstm: bool = False,
    ) -> None:
        """Initialise the summer rabbit hutch environment.

        Parameters
        ----------
        num_days : int
            Episode length in days.
        lstm_model : RabbitHutchLSTM | None
            Pre-instantiated (and optionally pre-trained) LSTM dynamics model.
            A fresh, untrained model is created when ``None``.
        cost_parameters_dict : dict
            Override default cost parameters.
        noisy_disturbance : bool
            Add Gaussian noise to the synthetic disturbance profile.
        pretrain_lstm : bool
            If ``True`` and *lstm_model* is ``None``, the freshly created LSTM
            is pre-trained on synthetic data before first use.
        """
        super().__init__()

        if lstm_model is None:
            lstm_model = RabbitHutchLSTM()
            if pretrain_lstm:
                transitions = RabbitHutchLSTM.generate_synthetic_transitions(
                    num_steps=8000
                )
                lstm_model.train_on_transitions(transitions, epochs=200)
        self.lstm_model = lstm_model
        self.lstm_hidden: tuple | None = None  # LSTM recurrent state

        # observation and action spaces
        x_lb = RabbitHutchLSTM.get_x_min()
        x_ub = RabbitHutchLSTM.get_x_max()
        self.observation_space = Box(x_lb, x_ub, (self.nx,), np.float64)
        self.action_space = Box(
            RabbitHutchLSTM.get_u_min(),
            RabbitHutchLSTM.get_u_max(),
            (self.nu,),
            np.float64,
        )
        self._x_lb = x_lb
        self._x_ub = x_ub

        # cost parameters
        self.c_u: np.ndarray = cost_parameters_dict.get(
            "c_u", np.array([1.0, 2.0, 0.5])
        )  # energy penalty per control input
        self.c_dy: float = cost_parameters_dict.get(
            "c_dy", 0.0
        )  # not used, kept for API compatibility
        self.w_y: np.ndarray = cost_parameters_dict.get(
            "w_y", np.full(self.nx, 1e3)
        )  # constraint violation penalty

        self.num_days = num_days
        self.noisy_disturbance = noisy_disturbance
        self.yield_step: int = self.steps_per_day * num_days - 1

        # du constraint (10 % of max per step)
        self.du_lim: np.ndarray = RabbitHutchLSTM.get_du_lim()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_disturbance(self) -> npt.NDArray[np.floating]:
        """Disturbance vector at the current time step."""
        return self.disturbance_profile[:, self.step_counter]

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Reset the environment to summer initial conditions.

        Parameters
        ----------
        seed : int | None
            RNG seed (forwarded to parent).
        options : dict | None
            Reserved; not used.

        Returns
        -------
        tuple
            Initial state and an empty info dict.
        """
        super().reset(seed=seed, options=options)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

        # comfortable summer indoor starting conditions
        self.x = np.array([24.0, 55.0, 800.0, 5.0])
        self.lstm_hidden = None  # reset recurrent state

        total_steps = self.steps_per_day * (self.num_days + 1)
        self.disturbance_profile = self._generate_summer_disturbances(total_steps)
        self.step_counter = 0
        self.previous_action = np.zeros(self.nu)

        assert self.observation_space.contains(
            self.x
        ), f"Initial state outside observation space: {self.x}"
        return self.x.copy(), {}

    def get_stage_cost(
        self,
        state: npt.NDArray[np.floating],
        action: npt.NDArray[np.floating],
    ) -> float:
        """Compute the stage cost (control energy + constraint violations).

        Parameters
        ----------
        state : np.ndarray, shape (nx,)
        action : np.ndarray, shape (nu,)

        Returns
        -------
        float
        """
        y_min = RabbitHutchLSTM.get_y_min_summer()
        y_max = RabbitHutchLSTM.get_y_max_summer()
        y_range = RabbitHutchLSTM.get_y_range_summer()

        cost = 0.0
        # penalise control energy
        cost += float(np.dot(self.c_u, action))
        # penalise constraint violations (normalised)
        viol_low = np.maximum(0.0, (y_min - state) / y_range)
        viol_high = np.maximum(0.0, (state - y_max) / y_range)
        cost += float(np.dot(self.w_y, viol_low + viol_high))
        return cost

    def step(
        self,
        action: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Advance the environment by one time step.

        Parameters
        ----------
        action : np.ndarray, shape (nu,)

        Returns
        -------
        tuple
            (observation, reward, truncated, terminated, info)
        """
        u = np.asarray(action, dtype=np.float64).reshape(self.nu)
        assert self.action_space.contains(u), f"Invalid action in `step`: {u}"

        x = self.x
        cost = self.get_stage_cost(x, u)
        d = self.current_disturbance

        # LSTM one-step prediction
        x_next_raw, self.lstm_hidden = self.lstm_model.predict(
            x.astype(np.float32),
            u.astype(np.float32),
            d.astype(np.float32),
            self.lstm_hidden,
        )
        x_next = np.clip(x_next_raw.astype(np.float64), self._x_lb, self._x_ub)

        assert self.observation_space.contains(
            x_next
        ), f"Invalid next state in `step`: {x_next}"

        self.previous_action = u
        self.x = x_next.copy()
        truncated = self.step_counter == self.yield_step
        self.step_counter += 1
        return x_next, float(cost), truncated, False, {}

    # ------------------------------------------------------------------
    # Disturbance helpers
    # ------------------------------------------------------------------

    def get_current_disturbance(self, length: int) -> npt.NDArray[np.floating]:
        """Return the disturbance window ``[t, t + length)``.

        Parameters
        ----------
        length : int
            Number of steps to return.

        Returns
        -------
        np.ndarray, shape (nd, length)
        """
        return self.disturbance_profile[
            :, self.step_counter : self.step_counter + length
        ]

    def _generate_summer_disturbances(self, num_steps: int) -> npt.NDArray[np.floating]:
        """Generate a synthetic summer disturbance profile.

        The profile represents a typical hot summer day:
        - Outdoor temperature peaks at ~40 °C around 14:00.
        - Solar radiation follows a half-sine during daylight hours.
        - Outdoor humidity is anticorrelated with temperature.

        Parameters
        ----------
        num_steps : int
            Number of 15-minute steps.

        Returns
        -------
        np.ndarray, shape (nd, num_steps)
        """
        t = np.linspace(0.0, 2.0 * np.pi * (num_steps / self.steps_per_day), num_steps)

        # outdoor temperature: mean 34 °C, amplitude ±8 °C, peaks around 14:00
        outdoor_temp = 34.0 + 8.0 * np.sin(t - np.pi / 2)
        if self.noisy_disturbance:
            outdoor_temp += self.np_random.normal(0.0, 1.0, num_steps)

        # solar radiation: 0 at night, peak ~900 W/m² at noon
        solar_rad = np.maximum(0.0, 900.0 * np.sin(t))
        if self.noisy_disturbance:
            solar_rad += self.np_random.normal(0.0, 20.0, num_steps)
            solar_rad = np.maximum(0.0, solar_rad)

        # outdoor humidity: anticorrelated with temperature
        outdoor_hum = 60.0 - 20.0 * np.sin(t - np.pi / 2)
        outdoor_hum = np.clip(outdoor_hum, 20.0, 90.0)
        if self.noisy_disturbance:
            outdoor_hum += self.np_random.normal(0.0, 3.0, num_steps)
            outdoor_hum = np.clip(outdoor_hum, 20.0, 90.0)

        return np.vstack([outdoor_temp, solar_rad, outdoor_hum])

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def get_cost_parameters(self) -> dict:
        """Return the cost parameters dict."""
        return {"c_u": self.c_u, "c_dy": self.c_dy, "w_y": self.w_y}
