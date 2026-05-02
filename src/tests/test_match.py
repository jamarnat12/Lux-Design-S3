import jax
import jax.numpy as jnp
import numpy as np
from orbit_wars.env import OrbitWarsEnv
from orbit_wars.params import EnvParams
from kits.python.bot import agent

def run_match():
    env = OrbitWarsEnv()
    params = env.default_params
    key = jax.random.PRNGKey(1337)

    obs_dict, state = env.reset(key, params)

    for i in range(100):
        # Prepare actions for both players
        all_actions = []
        for p in range(2):
            # Convert JAX obs to dict format for bot
            player_obs = {
                "planets": np.array(obs_dict.planets).tolist(),
                "fleets": np.array(obs_dict.fleets).tolist(),
                "player": p,
                "steps": int(state.steps)
            }
            moves = agent(player_obs)

            # Convert moves to JAX array [max_launches, 3]
            jax_moves = jnp.zeros((10, 3)) # Assume max 10 launches per turn
            for m_idx, m in enumerate(moves[:10]):
                jax_moves = jax_moves.at[m_idx].set(jnp.array(m))
            all_actions.append(jax_moves)

        # Pad to 4 players
        for _ in range(2):
            all_actions.append(jnp.zeros((10, 3)))

        action_tensor = jnp.stack(all_actions) # [4, 10, 3]

        obs_dict, state, reward, done, truncated, info = env.step(key, state, action_tensor, params)

        if i % 10 == 0:
            active_fleets = jnp.sum(state.fleets_mask)
            p0_ships = jnp.sum(jnp.where(state.planets.owner == 0, state.planets.ships, 0))
            p1_ships = jnp.sum(jnp.where(state.planets.owner == 1, state.planets.ships, 0))
            print(f"Turn {i}: P0 ships={p0_ships}, P1 ships={p1_ships}, Active fleets={active_fleets}")

        if done:
            break

    print("Match finished.")
    print("Final rewards:", reward)

if __name__ == "__main__":
    run_match()
