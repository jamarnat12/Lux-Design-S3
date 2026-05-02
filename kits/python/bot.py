import numpy as np

def agent(obs):
    # Process observations
    planets = obs.get("planets", [])
    player = obs.get("player", 0)

    # Simple heuristic:
    # 1. Identify my planets
    # 2. Identify neutral planets
    # 3. For each of my planets, if it has enough ships, send a fleet to the nearest neutral planet

    my_planets = [p for p in planets if p[1] == player]
    neutral_planets = [p for p in planets if p[1] == -1]
    enemy_planets = [p for p in planets if p[1] != -1 and p[1] != player]

    moves = []

    for my_p in my_planets:
        my_id, _, my_x, my_y, _, my_ships, _ = my_p

        if my_ships > 20:
            # Find closest neutral or enemy planet
            targets = neutral_planets + enemy_planets
            if not targets:
                continue

            best_target = None
            min_dist = float('inf')

            for t in targets:
                t_id, t_owner, t_x, t_y, t_radius, t_ships, t_prod = t
                dist = np.sqrt((my_x - t_x)**2 + (my_y - t_y)**2)

                # Weight by production and ships
                score = dist / (t_prod + 1)

                if score < min_dist:
                    min_dist = score
                    best_target = t

            if best_target:
                t_id, t_owner, t_x, t_y, t_radius, t_ships, t_prod = best_target

                # Estimate travel time
                dist = np.sqrt((my_x - t_x)**2 + (my_y - t_y)**2)
                # Ships we are sending (roughly half)
                send_ships = int(my_ships // 2)
                speed = 1.0 + (6.0 - 1.0) * (np.log(max(send_ships, 1)) / np.log(1000.0)) ** 1.5
                travel_time = dist / max(speed, 0.1)

                # Target position at arrival (very simple prediction)
                # If target is orbiting, it will move.
                # For simplicity, we just use current position but a better bot would predict.

                angle = np.arctan2(t_y - my_y, t_x - my_x)

                # Check for sun collision
                # Sun at (50, 50) with radius 10
                # Vector from my planet to target
                vx = t_x - my_x
                vy = t_y - my_y
                # Vector from my planet to sun
                sx = 50.0 - my_x
                sy = 50.0 - my_y

                mag_sq = vx*vx + vy*vy
                if mag_sq > 0:
                    t_sun = (sx*vx + sy*vy) / mag_sq
                    t_sun = np.clip(t_sun, 0.0, 1.0)
                    closest_sun_x = my_x + t_sun * vx
                    closest_sun_y = my_y + t_sun * vy
                    dist_to_sun = np.sqrt((50.0 - closest_sun_x)**2 + (50.0 - closest_sun_y)**2)

                    if dist_to_sun < 11.0: # Sun radius + small buffer
                        # Try to aim slightly away? Or just don't send.
                        # For now, just skip if it hits the sun.
                        continue

                moves.append([int(my_id), float(angle), int(send_ships)])

    return moves
