# Agent for summer rabbit hutch MPC control.

from __future__ import annotations

import casadi as cs
import numpy as np
from csnlp import Solution
from mpcrl import Agent

from rabbit_hutch.env import RabbitHutchEnv
from rabbit_hutch.lstm_model import RabbitHutchLSTM


class RabbitHutchAgent(Agent):
    """MPC agent for the summer rabbit hutch environment.

    At each time step the agent:

    1. Reads the physical state ``x`` (shape ``(nx,)``) returned by the
       environment, plus the current LSTM hidden / cell state stored on
       ``env.lstm_hidden``.
    2. Builds the augmented state ``z = [x ; h₁ ; … ; hₗ ; c₁ ; … ; cₗ]``
       that the MPC expects.
    3. Updates the disturbance and comfort-bound parameters on the MPC.
    4. Solves the MPC and returns the first optimal control action.

    Solve times are accumulated per episode for diagnostics.
    """

    solve_times: list[list[float]] = []
    _time_step_computation_time: list[float] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_augmented_state(
        x: np.ndarray,
        lstm_hidden: tuple | None,
        hs: int,
        nl: int,
    ) -> np.ndarray:
        """Concatenate physical state with LSTM hidden / cell vectors."""
        if lstm_hidden is None:
            h_flat = np.zeros(nl * hs)
            c_flat = np.zeros(nl * hs)
        else:
            h_tensors, c_tensors = lstm_hidden
            h_flat = h_tensors.detach().numpy().reshape(-1)
            c_flat = c_tensors.detach().numpy().reshape(-1)
        return np.concatenate([x, h_flat, c_flat])

    def _set_mpc_parameters(self, env: RabbitHutchEnv) -> None:
        """Update disturbance window and comfort-bound parameters."""
        N = self.V.prediction_horizon
        d_horizon = env.get_current_disturbance(N + 1)  # (nd, N+1)
        self.fixed_parameters["d"] = d_horizon[:, :-1]  # (nd, N)

        y_min = RabbitHutchLSTM.get_y_min_summer()
        y_max = RabbitHutchLSTM.get_y_max_summer()
        for k in range(N + 1):
            self.fixed_parameters[f"y_min_{k}"] = y_min
            self.fixed_parameters[f"y_max_{k}"] = y_max

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_episode_start(
        self, env: RabbitHutchEnv, episode: int, state: np.ndarray
    ) -> None:
        """Store env reference, initialise parameters, start new episode."""
        self._env: RabbitHutchEnv = env
        self.solve_times.append([])
        self._set_mpc_parameters(env)
        return super().on_episode_start(env, episode, state)

    def on_env_step(
        self, env: RabbitHutchEnv, episode: int, timestep: int
    ) -> None:
        """Update MPC parameters after each environment step."""
        self._set_mpc_parameters(env)
        return super().on_env_step(env, episode, timestep)

    def on_timestep_end(
        self, env: RabbitHutchEnv, episode: int, timestep: int
    ) -> None:
        self.solve_times[-1].append(sum(self._time_step_computation_time))
        self._time_step_computation_time = []
        return super().on_timestep_end(env, episode, timestep)

    # ------------------------------------------------------------------
    # state_value: translate physical state → augmented state for MPC
    # ------------------------------------------------------------------

    def state_value(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        vals0=None,
        action_space=None,
        **kwargs,
    ) -> tuple[cs.DM, Solution]:
        """Augment the physical state with LSTM hidden states, then call MPC.

        Parameters
        ----------
        state : np.ndarray, shape (nx,)
            Physical state from the environment observation.
        deterministic : bool
            Whether to suppress exploration noise.

        Returns
        -------
        tuple
            ``(action, solution)`` from the MPC solve.
        """
        hs = self.V.hs
        nl = self.V.nl
        env = getattr(self, "_env", None)
        lstm_hidden = env.lstm_hidden if env is not None else None
        z = self._build_augmented_state(state, lstm_hidden, hs, nl)

        action, sol = super().state_value(
            z, deterministic, vals0, action_space, **kwargs
        )
        if "t_wall_total" in sol.stats:
            self._time_step_computation_time.append(sol.stats["t_wall_total"])
        return action, sol
