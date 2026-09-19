def format_time(total_time: float) -> str:
    """Formats a time duration given in seconds into a human-readable string."""
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds:.2f}s ({total_time:.2f} seconds)"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds:.2f}s ({total_time:.2f} seconds)"
    else:
        time_str = f"{seconds:.2f} seconds"

    return time_str
