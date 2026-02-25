"""nominal_rabbit_hutch.py
Summer rabbit hutch climate control using an LSTM dynamics model and MPC.

Usage
-----
    python nominal_rabbit_hutch.py

The script:
1. Builds and pre-trains an LSTM dynamics model on synthetic summer data.
2. Wraps the model in a gym environment (RabbitHutchEnv).
3. Constructs a CasADi MPC (RabbitHutchMpc) that uses the LSTM symbolically.
4. Runs a short evaluation episode and prints / plots the results.
"""

import logging

import numpy as np
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes

from agents.rabbit_hutch_agent import RabbitHutchAgent
from mpcs.rabbit_hutch_mpc import RabbitHutchMpc
from rabbit_hutch.env import RabbitHutchEnv
from rabbit_hutch.lstm_model import RabbitHutchLSTM

np_random = np.random.default_rng(42)

STORE_DATA = False
PLOT = True

# ---------------------------------------------------------------------------
# 1. Build and pre-train the LSTM dynamics model
# ---------------------------------------------------------------------------
print("Pre-training LSTM on synthetic summer data …")
lstm_model = RabbitHutchLSTM(hidden_size=16, num_layers=1)
transitions = RabbitHutchLSTM.generate_synthetic_transitions(
    num_steps=8000, np_random=np_random
)
lstm_model.train_on_transitions(transitions, epochs=300, lr=1e-3, verbose=True)
print("LSTM pre-training complete.\n")

# ---------------------------------------------------------------------------
# 2. Create the gym environment
# ---------------------------------------------------------------------------
days = 1
episode_len = days * RabbitHutchEnv.steps_per_day

env = MonitorEpisodes(
    TimeLimit(
        RabbitHutchEnv(
            num_days=days,
            lstm_model=lstm_model,
            cost_parameters_dict={
                "c_u": np.array([1.0, 2.0, 0.5]),
                "w_y": 1e3 * np.ones(RabbitHutchEnv.nx),
            },
            noisy_disturbance=True,
        ),
        max_episode_steps=int(episode_len),
    )
)

# ---------------------------------------------------------------------------
# 3. Build the MPC
# ---------------------------------------------------------------------------
prediction_horizon = 6 * 4  # 6 hours ahead

mpc = RabbitHutchMpc(
    env=env,
    lstm_model=lstm_model,
    prediction_horizon=prediction_horizon,
    cost_parameters_dict={
        "c_u": np.array([1.0, 2.0, 0.5]),
        "w_y": 1e3 * np.ones(RabbitHutchEnv.nx),
    },
    constrain_control_rate=True,
)

# ---------------------------------------------------------------------------
# 4. Wrap agent in a logger and evaluate
# ---------------------------------------------------------------------------
agent = Log(
    RabbitHutchAgent(mpc, fixed_parameters={}),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 20},
    to_file=False,
    log_name="rabbit_hutch_nominal",
)

agent.evaluate(
    env=env,
    episodes=1,
    seed=1,
    raises=False,
)

# ---------------------------------------------------------------------------
# 5. Extract and display results
# ---------------------------------------------------------------------------
X = np.asarray(env.observations)    # (episodes, T+1, nx)
U = np.asarray(env.actions).squeeze(-1)  # (episodes, T, nu)
R = np.asarray(env.rewards)          # (episodes, T)

print(f"Total cost (lower is better): {R.sum(axis=1)}")
print(
    f"Average solve time per step: "
    f"{np.mean([t for ep in agent.solve_times for t in ep]):.4f} s"
)

# ---------------------------------------------------------------------------
# 6. Optional plot
# ---------------------------------------------------------------------------
if PLOT:
    try:
        import matplotlib.pyplot as plt

        state_labels = ["Temperature (°C)", "Humidity (%)", "CO₂ (ppm)", "NH₃ (ppm)"]
        ctrl_labels = ["Ventilation", "Cooling", "Shade"]
        dist_labels = ["Outdoor Temp (°C)", "Solar Rad (W/m²)", "Outdoor Hum (%)"]

        y_min = RabbitHutchLSTM.get_y_min_summer()
        y_max = RabbitHutchLSTM.get_y_max_summer()

        ep = 0  # plot first episode
        T = X.shape[1] - 1
        time_axis = np.arange(T) * (RabbitHutchEnv.ts / 3600.0)  # hours

        # — state trajectories —
        fig, axs = plt.subplots(
            RabbitHutchEnv.nx, 1, sharex=True, constrained_layout=True
        )
        fig.suptitle("Rabbit Hutch – State Trajectories (Summer)")
        for i, ax in enumerate(axs):
            ax.plot(time_axis, X[ep, :-1, i], label="state")
            ax.axhline(y_min[i], color="green", linestyle="--", label="min")
            ax.axhline(y_max[i], color="red", linestyle="--", label="max")
            ax.set_ylabel(state_labels[i])
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="upper right")

        # — control actions —
        fig2, axs2 = plt.subplots(
            RabbitHutchEnv.nu, 1, sharex=True, constrained_layout=True
        )
        fig2.suptitle("Rabbit Hutch – Control Actions (Summer)")
        u_min = RabbitHutchLSTM.get_u_min()
        u_max_arr = RabbitHutchLSTM.get_u_max()
        for i, ax in enumerate(axs2):
            ax.plot(time_axis, U[ep, :, i])
            ax.axhline(u_min[i], color="green", linestyle="--")
            ax.axhline(u_max_arr[i], color="red", linestyle="--")
            ax.set_ylabel(ctrl_labels[i])
        axs2[-1].set_xlabel("Time (h)")

        # — stage cost —
        fig3, ax3 = plt.subplots(1, 1, constrained_layout=True)
        ax3.plot(time_axis, R[ep])
        ax3.set_xlabel("Time (h)")
        ax3.set_ylabel("Stage cost")
        ax3.set_title("Rabbit Hutch – Stage Cost (Summer)")

        plt.show()
    except Exception as exc:
        print(f"Plotting skipped: {exc}")

if STORE_DATA:
    import pickle

    with open("rabbit_hutch_nominal.pkl", "wb") as fh:
        pickle.dump({"X": X, "U": U, "R": R}, fh)
    print("Data saved to rabbit_hutch_nominal.pkl")
