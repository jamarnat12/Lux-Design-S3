import jax
import jax.numpy as jnp
from orbit_wars.env import OrbitWarsEnv
from orbit_wars.params import EnvParams

def test_env():
    env = OrbitWarsEnv()
    params = env.default_params
    key = jax.random.PRNGKey(42)

    obs, state = env.reset(key, params)
    print("Initial state steps:", state.steps)
    print("Num planets:", len(state.planets.id))

    # Mock action: player 0 launches from planet 0
    # action shape: [num_players, max_launches, 3]
    # For now my env.py _process_launches is a placeholder, let's refine it.

    # Just step without actions for now
    action = jnp.zeros((2, 1, 3))
    obs, state, reward, done, truncated, info = env.step(key, state, action, params)
    print("Step 1 done. Ships on planet 0:", state.planets.ships[0])
    print("Production on planet 0:", state.planets.production[0])

if __name__ == "__main__":
    test_env()
