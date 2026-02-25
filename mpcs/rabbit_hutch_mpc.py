# MPC for the summer rabbit hutch using LSTM dynamics expressed in CasADi.
#
# Because the rabbit hutch has no first-principles model, the LSTM neural
# network (trained offline) is used as the prediction model inside the MPC.
# The LSTM cell equations are re-implemented symbolically in CasADi so that
# the NLP solver can differentiate through the multi-step predictions.
#
# Augmented state
# ---------------
# The recurrent hidden and cell states of the LSTM (h, c) are lifted into
# the MPC state so that the optimizer sees the full dynamics:
#
#   z_t = [ x_t ; h_t ; c_t ]   (dim = nx + 2 * num_layers * hidden_size)
#
# At each MPC call the current (h, c) from the environment are passed as the
# initial value of the augmented state.

from __future__ import annotations

import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc

from rabbit_hutch.env import RabbitHutchEnv
from rabbit_hutch.lstm_model import RabbitHutchLSTM


# ---------------------------------------------------------------------------
# CasADi LSTM cell helpers
# ---------------------------------------------------------------------------


def _sigmoid_cs(x: cs.SX) -> cs.SX:
    return 1.0 / (1.0 + cs.exp(-x))


def _lstm_cell_cs(
    inp: cs.SX,
    h: cs.SX,
    c: cs.SX,
    W_ih: np.ndarray,
    W_hh: np.ndarray,
    b_ih: np.ndarray,
    b_hh: np.ndarray,
) -> tuple[cs.SX, cs.SX]:
    """One LSTM cell step expressed as CasADi SX operations.

    Parameters
    ----------
    inp : cs.SX, shape (input_size, 1)
    h   : cs.SX, shape (hidden_size, 1)  — previous hidden state
    c   : cs.SX, shape (hidden_size, 1)  — previous cell state
    W_ih, W_hh, b_ih, b_hh              — weight matrices / vectors (numpy)

    Returns
    -------
    (h_new, c_new) as CasADi SX expressions.
    """
    hs = h.shape[0]
    W_ih_dm = cs.DM(W_ih)
    W_hh_dm = cs.DM(W_hh)
    b_ih_dm = cs.DM(b_ih.reshape(-1, 1))
    b_hh_dm = cs.DM(b_hh.reshape(-1, 1))

    gates = W_ih_dm @ inp + b_ih_dm + W_hh_dm @ h + b_hh_dm  # (4*hs, 1)

    i_gate = _sigmoid_cs(gates[0 * hs : 1 * hs])
    f_gate = _sigmoid_cs(gates[1 * hs : 2 * hs])
    g_gate = cs.tanh(gates[2 * hs : 3 * hs])
    o_gate = _sigmoid_cs(gates[3 * hs : 4 * hs])

    c_new = f_gate * c + i_gate * g_gate
    h_new = o_gate * cs.tanh(c_new)
    return h_new, c_new


def lstm_step_cs(
    x: cs.SX,
    u: cs.SX,
    d: cs.SX,
    h_list: list[cs.SX],
    c_list: list[cs.SX],
    weights: dict[str, np.ndarray],
    hidden_size: int,
    num_layers: int,
) -> tuple[cs.SX, list[cs.SX], list[cs.SX]]:
    """Full LSTM forward step (all layers) in CasADi.

    Returns
    -------
    (x_next, h_new_list, c_new_list)
    """
    inp = cs.vertcat(x, u, d)  # (nx+nu+nd, 1)
    h_new_list: list[cs.SX] = []
    c_new_list: list[cs.SX] = []
    current = inp
    for layer in range(num_layers):
        h_new, c_new = _lstm_cell_cs(
            current,
            h_list[layer],
            c_list[layer],
            weights[f"W_ih_{layer}"],
            weights[f"W_hh_{layer}"],
            weights[f"b_ih_{layer}"],
            weights[f"b_hh_{layer}"],
        )
        h_new_list.append(h_new)
        c_new_list.append(c_new)
        current = h_new  # last-layer output feeds into the next

    # linear output layer (residual)
    W_fc = cs.DM(weights["W_fc"])
    b_fc = cs.DM(weights["b_fc"].reshape(-1, 1))
    delta_x = W_fc @ current + b_fc
    x_next = x + delta_x
    return x_next, h_new_list, c_new_list


# ---------------------------------------------------------------------------
# RabbitHutchMpc
# ---------------------------------------------------------------------------


class RabbitHutchMpc(Mpc[cs.SX]):
    """Non-linear MPC for summer rabbit hutch control using LSTM dynamics.

    The LSTM hidden / cell states are lifted into the MPC state vector so
    that predictions are fully differentiable.  The physical state ``x``
    (temperature, humidity, CO₂, NH₃) is separated from the LSTM states
    in the objective and constraints.

    Parameters
    ----------
    env : RabbitHutchEnv
        The rabbit hutch environment (used for dimensions and bounds).
    lstm_model : RabbitHutchLSTM
        Pre-trained LSTM dynamics model.
    prediction_horizon : int
        MPC prediction horizon (number of 15-min steps).
    cost_parameters_dict : dict
        Override cost parameters.
    constrain_control_rate : bool
        Whether to add du constraints.
    """

    def __init__(
        self,
        env: RabbitHutchEnv,
        lstm_model: RabbitHutchLSTM,
        prediction_horizon: int = 6 * 4,
        cost_parameters_dict: dict = {},
        constrain_control_rate: bool = True,
    ) -> None:
        nx = env.nx
        nu = env.nu
        nd = env.nd
        hs = lstm_model.hidden_size
        nl = lstm_model.num_layers

        u_min = RabbitHutchLSTM.get_u_min()
        u_max = RabbitHutchLSTM.get_u_max()
        du_lim = RabbitHutchLSTM.get_du_lim()

        if not cost_parameters_dict:
            cost_parameters_dict = env.get_cost_parameters()
        c_u: np.ndarray = cost_parameters_dict.get("c_u", np.array([1.0, 2.0, 0.5]))
        w_y: np.ndarray = cost_parameters_dict.get("w_y", np.full(nx, 1e3))

        # augmented state dimension: x + h (all layers) + c (all layers)
        nz = nx + 2 * nl * hs  # total augmented state size

        nlp = Nlp[cs.SX](debug=False)
        super().__init__(nlp, prediction_horizon=prediction_horizon)
        N = self.prediction_horizon

        # extract trained weights for CasADi
        weights = lstm_model.get_casadi_weights()

        # ------------------------------------------------------------------
        # Variables
        # ------------------------------------------------------------------
        z, _ = self.state("z", nz)  # augmented state (x, h, c)
        u, _ = self.action("u", nu, lb=u_min.reshape(-1, 1), ub=u_max.reshape(-1, 1))
        self.disturbance("d", nd)
        s, _, _ = self.variable("s", (nx, N + 1), lb=0)  # slack variables

        # ------------------------------------------------------------------
        # Dynamics: LSTM cell unrolled symbolically
        # ------------------------------------------------------------------
        def augmented_dynamics(z_k: cs.SX, u_k: cs.SX, d_k: cs.SX) -> cs.SX:
            x_k = z_k[:nx]
            # unpack h and c for each layer
            h_list = [
                z_k[nx + layer * hs : nx + (layer + 1) * hs]
                for layer in range(nl)
            ]
            c_list = [
                z_k[nx + nl * hs + layer * hs : nx + nl * hs + (layer + 1) * hs]
                for layer in range(nl)
            ]
            x_next, h_new_list, c_new_list = lstm_step_cs(
                x_k, u_k, d_k, h_list, c_list, weights, hs, nl
            )
            z_next = cs.vertcat(x_next, *h_new_list, *c_new_list)
            return z_next

        self.set_dynamics(augmented_dynamics, n_in=3, n_out=1)

        # ------------------------------------------------------------------
        # Constraints on the physical state x
        # ------------------------------------------------------------------
        y_min_summer = RabbitHutchLSTM.get_y_min_summer()
        y_max_summer = RabbitHutchLSTM.get_y_max_summer()
        y_range = RabbitHutchLSTM.get_y_range_summer()

        for k in range(N + 1):
            y_min_k = self.parameter(f"y_min_{k}", (nx, 1))
            y_max_k = self.parameter(f"y_max_{k}", (nx, 1))
            x_k = z[:nx, k]  # physical state at step k
            self.constraint(
                f"y_min_{k}", x_k, ">=", y_min_k - s[:, k] / y_range.reshape(-1, 1)
            )
            self.constraint(
                f"y_max_{k}", x_k, "<=", y_max_k + s[:, k] / y_range.reshape(-1, 1)
            )

        if constrain_control_rate:
            for k in range(1, N):
                self.constraint(f"du_min_{k}", u[:, k] - u[:, k - 1], "<=", du_lim)
                self.constraint(f"du_max_{k}", u[:, k] - u[:, k - 1], ">=", -du_lim)

        # ------------------------------------------------------------------
        # Objective
        # ------------------------------------------------------------------
        obj: cs.SX = cs.SX.zeros(1)
        for k in range(N):
            # control energy cost
            for j in range(nu):
                obj = obj + c_u[j] * u[j, k]
            # slack (constraint violation) penalty
            obj = obj + cs.dot(w_y, s[:, k])
        # terminal slack penalty
        obj = obj + cs.dot(w_y, s[:, N])
        self.minimize(obj)

        # ------------------------------------------------------------------
        # Solver
        # ------------------------------------------------------------------
        opts = {
            "expand": True,
            "show_eval_warnings": False,
            "warn_initial_bounds": True,
            "print_time": False,
            "record_time": True,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            "ipopt": {
                "sb": "yes",
                "print_level": 0,
                "max_iter": 2000,
                "print_user_options": "yes",
                "print_options_documentation": "no",
                "nlp_scaling_method": "gradient-based",
                "nlp_scaling_max_gradient": 10,
            },
        }
        self.init_solver(opts, solver="ipopt")

        # store dimensions for the agent
        self.nx_phys = nx
        self.hs = hs
        self.nl = nl
        self.nz = nz
        self._y_min_summer = y_min_summer
        self._y_max_summer = y_max_summer
