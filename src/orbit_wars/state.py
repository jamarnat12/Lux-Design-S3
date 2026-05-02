import chex
from flax import struct

@struct.dataclass
class PlanetState:
    id: int
    owner: int # 0-3, or -1 for neutral
    x: float
    y: float
    radius: float
    ships: int
    production: int
    orbital_radius: float
    initial_angle: float
    angular_velocity: float
    is_comet: bool

@struct.dataclass
class FleetState:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int
    speed: float

@struct.dataclass
class CometGroup:
    planet_ids: chex.Array # IDs of planets that are comets in this group
    paths: chex.Array # (4, path_length, 2)
    path_index: int

@struct.dataclass
class EnvState:
    planets: PlanetState
    fleets: FleetState
    fleets_mask: chex.Array # Bool mask for active fleets
    comet_groups: CometGroup
    steps: int
    next_fleet_id: int
    next_planet_id: int

@struct.dataclass
class EnvObs:
    planets: chex.Array # List of [id, owner, x, y, radius, ships, production]
    fleets: chex.Array # List of [id, owner, x, y, angle, from_planet_id, ships]
    player: int
    comet_planet_ids: chex.Array
    comets: chex.Array # Comet group data
    remainingOverageTime: float
