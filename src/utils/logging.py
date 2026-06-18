import logging
import sys


def setup_logging(level: str = "INFO"):
    """
    Configures the root logger with a standard format.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set lower level for noisy libraries if needed
    logging.getLogger("httpx").setLevel(logging.WARNING)
