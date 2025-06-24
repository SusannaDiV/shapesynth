import signal
from contextlib import contextmanager
import time

@contextmanager
def timeout(seconds, message="Operation timed out"):
    def signal_handler(signum, frame):
        raise TimeoutError(message)
    
    # Set the signal handler and a timeout
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0) 