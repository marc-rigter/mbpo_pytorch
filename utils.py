import numpy as np

def compute_traj_errors(env, observations, actions, rewards, sim_states, num_steps=[1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250, 300, 350, 400, 500]):
    """
        Observations and actions are [n_traj X traj_len X dim]
        Rewards is [n_traj X traj_len]
    """

    # loop through the observations and actions
    if not hasattr(env, "set_state"):
        return None, None
    
    metrics = dict()

    max_steps = observations.shape[1] - 1
    if max_steps not in num_steps and max_steps > 0:
        num_steps.append(max_steps)

    # compute open-loop predictions for different horizons
    for num_step in num_steps:
        obs_errors = []
        rew_errors = []
        
        if num_step > max_steps:
            continue

        # loop over each imagined trajectory
        for ep_obs, ep_act, ep_rew, ep_sim_state in zip(observations, actions, rewards, sim_states):

            # compute prediction error from initial state
            init_t = 0
            init_sim_state = ep_sim_state[init_t, :]

            # set initial simulator state
            env.reset()
            env.set_state(init_sim_state)

            # loop over the next num_step steps
            for k in range(num_step):
                a = ep_act[init_t + k]
                next_s_actual, r_actual, _, _, _ = env.step(a)

            # compute error in open-loop state prediction
            next_s_pred = ep_obs[init_t+num_step, :]
            obs_error = np.square(next_s_pred - next_s_actual).mean()
            obs_errors.append(obs_error)

            # only compute reward error for first step
            if num_step == 1:
                r_pred = ep_rew[init_t]
                r_error = np.square(r_pred - r_actual).mean()
                rew_errors.append(r_error)

        obs_errors = np.array(obs_errors)
        rew_errors = np.array(rew_errors)

        metrics.update({
            f"errors/dynamics_mse_{num_step:04}_step": obs_errors.mean(),
            f"errors/dynamics_mse_std_{num_step:04}_step": obs_errors.std(),
            f"errors/dynamics_mse_max_{num_step:04}_step": obs_errors.max(),
            f"errors/dynamics_mse_min_{num_step:04}_step": obs_errors.min(),
        })

        if num_step == 1:
            metrics.update({
                "errors/reward_mse": rew_errors.mean(),
                "errors/reward_mse_std": rew_errors.std(),
                "errors/reward_mse_max": rew_errors.max(),
                "errors/reward_mse_min": rew_errors.min(),
            })

    return metrics