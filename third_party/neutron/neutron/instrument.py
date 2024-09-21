import functools
import triton

from triton._C.libneutron import neutron as libneutron
from typing import Optional

DEFAULT_PROFILE_NAME = "neutron"


def activate(session: Optional[int] = 0) -> None:
    """
    Activate the specified session.
    The profiling session will be active and data will be recorded.

    Args:
        session (int): The session ID of the profiling session. Defaults to 0 (the first session started.)

    Returns:
        None
    """
    if is_command_line() and session != 0:
        raise ValueError("Only one session can be activated when running from the command line.")
    libneutron.activate(session)


def deactivate(session: Optional[int] = 0) -> None:
    """
    Stop the specified session.
    The profiling session's data will still be in the memory, but no more data will be recorded.

    Args:
        session (int): The session ID of the profiling session. Defaults to 0 (the first session started.)

    Returns:
        None
    """
    if is_command_line() and session != 0:
        raise ValueError("Only one session can be deactivated when running from the command line.")
    libneutron.deactivate(session)
