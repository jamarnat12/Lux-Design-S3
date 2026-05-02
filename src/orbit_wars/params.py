import chex
from flax import struct

@struct.dataclass
class EnvParams:
    map_width: float = 100.0
    map_height: float = 100.0
    sun_pos: chex.Array = struct.field(default_factory=lambda: [50.0, 50.0])
    sun_radius: float = 10.0
    max_steps: int = 500

    # Planet params
    min_production: int = 1
    max_production: int = 5
    orbit_radius_limit: float = 50.0
    min_angular_velocity: float = 0.025
    max_angular_velocity: float = 0.05

    # Fleet params
    max_speed: float = 6.0

    # Comet params
    comet_speed: float = 4.0
    comet_radius: float = 1.0
    comet_spawn_steps: chex.Array = struct.field(default_factory=lambda: [50, 150, 250, 350, 450])

    # Game rules
    num_players: int = 2 # or 4
    init_ships_home: int = 10
