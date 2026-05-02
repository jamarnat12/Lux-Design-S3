import functools
from typing import Any, Dict, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from jax import lax

from orbit_wars.params import EnvParams
from orbit_wars.state import EnvObs, EnvState, PlanetState, FleetState, CometGroup

class OrbitWarsEnv(environment.Environment):
    def __init__(self, max_planets: int = 44, max_fleets: int = 1000):
        super().__init__()
        self.max_planets = max_planets
        self.max_fleets = max_fleets

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: Optional[EnvParams] = None,
    ) -> Tuple[EnvObs, EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        if params is None:
            params = self.default_params
        obs, state, reward, done, truncated, info = self.step_env(
            key, state, action, params
        )
        return obs, state, reward, done, truncated, info

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Array, # [num_players, max_launches, 3] where 3 is [from_id, angle, ships]
        params: EnvParams,
    ) -> Tuple[EnvObs, EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:

        # 1. Comet expiration
        # (Implicitly handled by movement logic removing planets that leave the board)

        # 2. Comet spawning
        state = self._spawn_comets(state, params)

        # 3. Fleet launch
        state = self._process_launches(state, action, params)

        # 4. Production
        state = self._process_production(state, params)

        # 5. Fleet movement & Planet Collision detection
        state, collided_fleets = self._move_fleets(state, params)

        # 6. Planet rotation & Comet movement
        state, swept_fleets = self._move_planets_and_comets(state, params)

        # 7. Combat resolution
        state = self._resolve_combat(state, collided_fleets | swept_fleets, params)

        # Update steps
        state = state.replace(steps=state.steps + 1)

        # Check termination
        done = state.steps >= params.max_steps
        # Elimination check: check if any player has planets or fleets
        def player_has_assets(p_idx):
            has_planet = jnp.any(state.planets.owner == p_idx)
            has_fleet = jnp.any(state.fleets_mask & (state.fleets.owner == p_idx))
            return has_planet | has_fleet

        has_assets = jax.vmap(player_has_assets)(jnp.arange(4))
        num_active = jnp.sum(has_assets)
        done = done | (num_active <= 1)

        reward = self._get_reward(state, params)

        return (
            self.get_obs(state, params),
            state,
            reward,
            done,
            False,
            {}
        )

    def _spawn_comets(self, state: EnvState, params: EnvParams) -> EnvState:
        should_spawn = jnp.any(jnp.array(params.comet_spawn_steps) == state.steps)

        def do_spawn(state):
            # For simplicity, create 4 comets entering from corners
            new_ids = jnp.arange(4) + state.next_planet_id
            # Very simple linear paths for this implementation
            # In a real environment, these would be pre-calculated ellipses
            comet_planets = PlanetState(
                id=new_ids,
                owner=jnp.full(4, -1),
                x=jnp.array([0.0, 100.0, 0.0, 100.0]),
                y=jnp.array([0.0, 0.0, 100.0, 100.0]),
                radius=jnp.full(4, params.comet_radius),
                ships=jnp.full(4, 20), # Random starting ships
                production=jnp.full(4, 1),
                orbital_radius=jnp.full(4, -1.0), # Use -1 to indicate comet/linear path
                initial_angle=jnp.zeros(4),
                angular_velocity=jnp.zeros(4),
                is_comet=jnp.full(4, True)
            )
            # Find first 4 inactive planet slots (assuming max_planets is large enough)
            # Here we just append or use next_planet_id logic
            # For JAX, we'll just overwrite from a fixed comet buffer or similar
            # Simplified: we assume last 4 slots in planets array are for the most recent comet group
            idx = (self.max_planets - 4)
            return state.replace(
                planets=jax.tree_util.tree_map(lambda x, y: x.at[idx:idx+4].set(y), state.planets, comet_planets),
                next_planet_id=state.next_planet_id + 4
            )

        return jax.lax.cond(should_spawn, do_spawn, lambda s: s, state)

    def _process_launches(self, state: EnvState, action: chex.Array, params: EnvParams) -> EnvState:
        # action: [num_players, max_launches, 3] -> [from_id, angle, ships]

        def player_launches(carry, player_idx):
            state = carry
            player_actions = action[player_idx] # [max_launches, 3]

            def single_launch(carry, act):
                state = carry
                from_id, angle, num_ships = act

                # Check if player owns the planet and has enough ships
                # For JAX, we need to handle this without dynamic indexing where possible
                # or use valid_launch mask
                planet_idx = jnp.where(state.planets.id == from_id, size=1, fill_value=-1)[0][0]
                valid_launch = (planet_idx != -1) & (state.planets.owner[planet_idx] == player_idx) & (state.planets.ships[planet_idx] >= num_ships) & (num_ships > 0)

                # Update planet ships
                new_planet_ships = jnp.where(valid_launch, state.planets.ships[planet_idx] - num_ships, state.planets.ships[planet_idx])
                state = state.replace(planets=state.planets.replace(ships=state.planets.ships.at[planet_idx].set(new_planet_ships)))

                # Create fleet
                fleet_idx = jnp.argmin(state.fleets_mask)
                can_add_fleet = valid_launch & (~state.fleets_mask[fleet_idx])

                p_x = state.planets.x[planet_idx]
                p_y = state.planets.y[planet_idx]
                p_r = state.planets.radius[planet_idx]

                # Speed calculation
                log_ships = jnp.log(jnp.maximum(num_ships, 1))
                speed = 1.0 + (params.max_speed - 1.0) * jnp.power(log_ships / jnp.log(1000.0), 1.5)
                speed = jnp.clip(speed, 1.0, params.max_speed)

                new_fleet = FleetState(
                    id=state.next_fleet_id,
                    owner=player_idx,
                    x=p_x + jnp.cos(angle) * (p_r + 0.1),
                    y=p_y + jnp.sin(angle) * (p_r + 0.1),
                    angle=angle,
                    from_planet_id=from_id,
                    ships=num_ships.astype(jnp.int32),
                    speed=speed
                )

                state = state.replace(
                    fleets=jax.tree_util.tree_map(lambda x, y: x.at[fleet_idx].set(y), state.fleets, new_fleet),
                    fleets_mask=state.fleets_mask.at[fleet_idx].set(can_add_fleet),
                    next_fleet_id=state.next_fleet_id + 1
                )
                return state, None

            state, _ = jax.lax.scan(single_launch, state, player_actions)
            return state, None

        state, _ = jax.lax.scan(player_launches, state, jnp.arange(4))
        return state

    def _process_production(self, state: EnvState, params: EnvParams) -> EnvState:
        owned_mask = state.planets.owner != -1
        new_ships = state.planets.ships + jnp.where(owned_mask, state.planets.production, 0)
        return state.replace(planets=state.planets.replace(ships=new_ships))

    def _move_fleets(self, state: EnvState, params: EnvParams) -> Tuple[EnvState, chex.Array]:
        # Move fleets and check for collisions (sun, OOB, planets)
        dx = jnp.cos(state.fleets.angle) * state.fleets.speed
        dy = jnp.sin(state.fleets.angle) * state.fleets.speed

        old_x, old_y = state.fleets.x, state.fleets.y
        new_x, new_y = old_x + dx, old_y + dy

        # Sun collision (distance from (50,50) to segment)
        sun_collided = self._check_segment_collision(old_x, old_y, new_x, new_y, params.sun_pos[0], params.sun_pos[1], params.sun_radius)

        # OOB collision
        oob = (new_x < 0) | (new_x > params.map_width) | (new_y < 0) | (new_y > params.map_height)

        # Planet collision
        # [max_fleets, max_planets]
        planet_collisions = jax.vmap(lambda px, py, pr: self._check_segment_collision(old_x, old_y, new_x, new_y, px, py, pr))(
            state.planets.x, state.planets.y, state.planets.radius
        ).T # -> [max_fleets, max_planets]

        any_planet_collision = jnp.any(planet_collisions, axis=1)

        # Update fleet positions
        state = state.replace(fleets=state.fleets.replace(x=new_x, y=new_y))

        # Mark destroyed fleets
        destroyed = sun_collided | oob | any_planet_collision
        state = state.replace(fleets_mask=state.fleets_mask & ~destroyed)

        # Return fleets queued for combat (only those that hit a planet)
        return state, planet_collisions & state.fleets_mask[:, None]

    def _check_segment_collision(self, ax, ay, bx, by, px, py, pr) -> chex.Array:
        # Distance from point (px, py) to segment (ax, ay) -> (bx, by)
        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay

        mag_sq = vx*vx + vy*vy
        t = (wx*vx + wy*vy) / jnp.maximum(mag_sq, 1e-6)
        t = jnp.clip(t, 0.0, 1.0)

        closest_x = ax + t * vx
        closest_y = ay + t * vy

        dist_sq = (px - closest_x)**2 + (py - closest_y)**2
        return dist_sq <= pr**2

    def _move_planets_and_comets(self, state: EnvState, params: EnvParams) -> Tuple[EnvState, chex.Array]:
        # Orbiting planets rotation
        new_angle = state.planets.initial_angle + state.planets.angular_velocity * (state.steps + 1)
        # Only for orbiting planets (orbital_radius > 0)
        is_orbiting = state.planets.orbital_radius > 0
        new_x = jnp.where(is_orbiting, params.sun_pos[0] + jnp.cos(new_angle) * state.planets.orbital_radius, state.planets.x)
        new_y = jnp.where(is_orbiting, params.sun_pos[1] + jnp.sin(new_angle) * state.planets.orbital_radius, state.planets.y)

        # Comet movement (simplified linear move towards center and out)
        is_comet = state.planets.is_comet
        # Target center (50, 50)
        dx = 50.0 - state.planets.x
        dy = 50.0 - state.planets.y
        dist = jnp.sqrt(dx*dx + dy*dy + 1e-6)
        # Move at comet_speed but avoid sun collision for now or let them hit it
        new_x = jnp.where(is_comet, state.planets.x + (dx/dist) * params.comet_speed, new_x)
        new_y = jnp.where(is_comet, state.planets.y + (dy/dist) * params.comet_speed, new_y)

        # Remove comets that are far out (expiration)
        far_out = (jnp.abs(new_x - 50) > 60) | (jnp.abs(new_y - 50) > 60)
        should_remove = is_comet & far_out
        new_owner = jnp.where(should_remove, -1, state.planets.owner)
        new_ships = jnp.where(should_remove, 0, state.planets.ships)

        state = state.replace(planets=state.planets.replace(x=new_x, y=new_y, owner=new_owner, ships=new_ships))

        # Swept fleets: check if fleet position is now inside a planet
        # [max_fleets, max_planets]
        d_sq = (state.fleets.x[:, None] - state.planets.x[None, :])**2 + (state.fleets.y[:, None] - state.planets.y[None, :])**2
        swept = (d_sq <= state.planets.radius[None, :]**2) & state.fleets_mask[:, None]

        # Remove swept fleets from board
        any_swept = jnp.any(swept, axis=1)
        state = state.replace(fleets_mask=state.fleets_mask & ~any_swept)

        return state, swept

    def _resolve_combat(self, state: EnvState, queued_combats: chex.Array, params: EnvParams) -> EnvState:
        # queued_combats: [max_fleets, max_planets] (bool)

        def resolve_planet_combat(planet_idx, state):
            # Fleets hitting this planet
            hitting_mask = queued_combats[:, planet_idx]

            # Sum ships per owner (assuming fixed max players for JAX)
            # Use 4 as max possible players since spec says 1v1 or 4p FFA
            ships_per_owner = jnp.zeros(4)

            def sum_ships(p_idx, ships_per_owner):
                val = jnp.sum(jnp.where(hitting_mask & (state.fleets.owner == p_idx), state.fleets.ships, 0))
                return ships_per_owner.at[p_idx].set(val)

            # Manually unroll or use a fixed loop for JAX
            ships_per_owner = sum_ships(0, ships_per_owner)
            ships_per_owner = sum_ships(1, ships_per_owner)
            ships_per_owner = sum_ships(2, ships_per_owner)
            ships_per_owner = sum_ships(3, ships_per_owner)

            # Largest and second largest
            sorted_indices = jnp.argsort(ships_per_owner)[::-1]
            largest_owner = sorted_indices[0]
            second_largest_owner = sorted_indices[1]

            largest_ships = ships_per_owner[largest_owner]
            second_largest_ships = ships_per_owner[second_largest_owner]

            survivor_ships = largest_ships - second_largest_ships
            survivor_owner = jnp.where(largest_ships > second_largest_ships, largest_owner, -1)

            # Resolve against garrison
            planet_owner = state.planets.owner[planet_idx]
            planet_ships = state.planets.ships[planet_idx]

            def same_owner_case():
                return state.planets.ships.at[planet_idx].set(planet_ships + survivor_ships), state.planets.owner

            def diff_owner_case():
                new_owner = jnp.where(survivor_ships > planet_ships, survivor_owner, planet_owner)
                new_ships = jnp.abs(survivor_ships - planet_ships)
                # If tied exactly, survivor_ships == planet_ships, new_ships = 0, owner remains same?
                # Spec says "If attackers exceed the garrison, the planet changes ownership"
                return state.planets.ships.at[planet_idx].set(new_ships), state.planets.owner.at[planet_idx].set(new_owner)

            new_ships, new_owners = lax.cond(
                (survivor_owner == planet_owner) & (survivor_owner != -1),
                same_owner_case,
                diff_owner_case
            )

            return state.replace(planets=state.planets.replace(ships=new_ships, owner=new_owners))

        # We need to iterate over all planets. Since JAX doesn't like dynamic updates in vmap,
        # we might need to compute all combat results first then apply.
        # But planets are independent for combat in a single turn.

        for i in range(self.max_planets):
            state = resolve_planet_combat(i, state)

        return state

    def get_obs(self, state: EnvState, params: EnvParams) -> EnvObs:
        # Construct observation for player 0 (standard Kagle format)
        return EnvObs(
            planets=jnp.stack([
                state.planets.id,
                state.planets.owner,
                state.planets.x,
                state.planets.y,
                state.planets.radius,
                state.planets.ships,
                state.planets.production
            ], axis=1),
            fleets=jnp.stack([
                state.fleets.id,
                state.fleets.owner,
                state.fleets.x,
                state.fleets.y,
                state.fleets.angle,
                state.fleets.from_planet_id,
                state.fleets.ships
            ], axis=1),
            player=0, # This would be vectorized for all players
            comet_planet_ids=jnp.array([]), # TODO
            comets=jnp.array([]), # TODO
            remainingOverageTime=60.0
        )

    def _get_reward(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        # Final score = total ships on owned planets + total ships in owned fleets
        def get_player_score(p_idx):
            planet_ships = jnp.sum(jnp.where(state.planets.owner == p_idx, state.planets.ships, 0))
            fleet_ships = jnp.sum(jnp.where(state.fleets_mask & (state.fleets.owner == p_idx), state.fleets.ships, 0))
            return planet_ships + fleet_ships

        # Use fixed size for rewards too
        scores = jax.vmap(get_player_score)(jnp.arange(4))
        # Mask out non-existent players if needed, but for now just return all 4
        return scores

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[EnvObs, EnvState]:
        key_p, key_o, key_s, key_h = jax.random.split(key, 4)

        # Symmetry: 5-10 symmetric groups of 4 = 20-40 planets
        num_groups = jax.random.randint(key_p, (), 5, 11)

        def gen_group(carry, i):
            key, next_id = carry
            k1, k2, k3, k4 = jax.random.split(key, 4)

            # Random position in Q1 (0-50, 0-50)
            x = jax.random.uniform(k1, (), minval=10, maxval=45)
            y = jax.random.uniform(k2, (), minval=10, maxval=45)

            prod = jax.random.randint(k3, (), params.min_production, params.max_production + 1)
            radius = 1.0 + jnp.log(prod.astype(jnp.float32))

            ships = jax.random.randint(k4, (), 5, 100)

            # Orbit logic
            dist = jnp.sqrt((x-50)**2 + (y-50)**2)
            is_orbiting = dist + radius < 50.0
            orbital_radius = jnp.where(is_orbiting, dist, 0.0)
            initial_angle = jnp.where(is_orbiting, jnp.arctan2(y-50, x-50), 0.0)
            ang_vel = jnp.where(is_orbiting, jax.random.uniform(k1, (), minval=params.min_angular_velocity, maxval=params.max_angular_velocity), 0.0)

            # Create 4 symmetric planets
            # Quad 1: (x, y), Quad 2: (100-x, y), Quad 3: (x, 100-y), Quad 4: (100-x, 100-y)
            group_planets = PlanetState(
                id=jnp.arange(4) + next_id,
                owner=jnp.full(4, -1),
                x=jnp.array([x, 100-x, x, 100-x]),
                y=jnp.array([y, y, 100-y, 100-y]),
                radius=jnp.full(4, radius),
                ships=jnp.full(4, ships),
                production=jnp.full(4, prod),
                orbital_radius=jnp.full(4, orbital_radius),
                initial_angle=jnp.array([
                    initial_angle, # Q1
                    jnp.arctan2(y-50, (100-x)-50), # Q2
                    jnp.arctan2((100-y)-50, x-50), # Q3
                    jnp.arctan2((100-y)-50, (100-x)-50) # Q4
                ]),
                angular_velocity=jnp.full(4, ang_vel),
                is_comet=jnp.full(4, False)
            )
            return (k1, next_id + 4), group_planets

        _, all_groups = jax.lax.scan(gen_group, (key_o, 0), jnp.arange(10))

        # Flatten all_groups
        planets = jax.tree_util.tree_map(lambda x: x.reshape(-1), all_groups)

        # Choose home planets group
        home_group_idx = jax.random.randint(key_h, (), 0, 10)

        # Players start on diagonally opposite planets (Q1 and Q4) -> index 0 and 3 in our group layout
        # (x, y) and (100-x, 100-y)
        planets = planets.replace(
            owner=planets.owner.at[home_group_idx*4 + 0].set(0).at[home_group_idx*4 + 3].set(1),
            ships=planets.ships.at[home_group_idx*4 + 0].set(params.init_ships_home).at[home_group_idx*4 + 3].set(params.init_ships_home)
        )

        state = EnvState(
            planets=planets,
            fleets=FleetState(
                id=jnp.zeros(self.max_fleets, dtype=jnp.int32),
                owner=jnp.zeros(self.max_fleets, dtype=jnp.int32),
                x=jnp.zeros(self.max_fleets),
                y=jnp.zeros(self.max_fleets),
                angle=jnp.zeros(self.max_fleets),
                from_planet_id=jnp.zeros(self.max_fleets, dtype=jnp.int32),
                ships=jnp.zeros(self.max_fleets, dtype=jnp.int32),
                speed=jnp.zeros(self.max_fleets)
            ),
            fleets_mask=jnp.zeros(self.max_fleets, dtype=jnp.bool),
            comet_groups=CometGroup(
                planet_ids=jnp.array([]),
                paths=jnp.array([]),
                path_index=0
            ),
            steps=0,
            next_fleet_id=0,
            next_planet_id=self.max_planets
        )

        return self.get_obs(state, params), state
