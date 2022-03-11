__all__ = ["linear_schedule"]

def linear_schedule(
    step: int, *, value_from: float, value_to: float, nsteps: int
) -> float:
    """Basic linear decay schedule

    Usage:
        schedule_fn = partial(linear_schedule, value_from=x, value_to=y, nsteps=z)
        current_value = schedule_fn(step)

    Args:
        step: Current step in schedule
        *: ignore other arguments
        value_from: Starting value
        value_to: Ending value
        nsteps: Number of steps (same reference as "step" argument) over which to decay
    Returns:
        Decayed value
    """
    t = min(max(0.0, step / (nsteps - 1)), 1.0)
    return value_from * (1.0 - t) + value_to * t
