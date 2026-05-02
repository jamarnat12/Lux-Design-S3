# Orbit Wars - Usage Guide

This project implements the Orbit Wars environment using JAX and a starter AI bot.

## Installation

Ensure you have the dependencies installed:

```bash
pip install jax jaxlib gymnax==0.0.8 flax chex numpy
```

## Running a Match

You can run a simulated match between two instances of the starter bot using the provided test script. This script will output the game state every 10 turns.

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 src/tests/test_match.py
```

## Bot Implementation

The starter bot is located at `kits/python/bot.py`. It follows the standard competition format:

```python
def agent(obs):
    # obs contains 'planets', 'fleets', 'player', etc.
    # returns a list of moves: [[from_id, angle, ships], ...]
    ...
    return moves
```

## Environment Details

The environment is implemented in `src/orbit_wars/env.py` using JAX for high-performance simulation. It supports:
- 4-fold symmetric map generation.
- Orbital mechanics (rotating planets).
- Continuous collision detection for fleets.
- Multi-player combat resolution.

## Next Steps

- **Improve Heuristics:** Edit `kits/python/bot.py` to implement more advanced strategies like predicting orbital positions or multi-planet coordination.
- **Training:** Since the environment is built with JAX, you can use it with Reinforcement Learning frameworks to train a bot.
