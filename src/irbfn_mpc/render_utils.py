def render_lookahead_point(lookahead_point, e):
    """
    Callback to render the lookahead point.
    """
    points = lookahead_point[:2][None]  # shape (1, 2)
    e.render_points(points, color=(0, 0, 128), size=2)


def render_local_plan(waypoints, current_index, e):
    """
    update waypoints being drawn by EnvRenderer
    """
    points = waypoints[current_index : current_index + 10, :2]
    e.render_lines(points, color=(0, 128, 0), size=1)

