import sys


class Tee:
    """Class to duplicate output to both stdout and a file"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()  # Force write to file immediately
            except (BrokenPipeError, ValueError, OSError):
                # Stream may be closing during interpreter shutdown.
                continue
    
    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except (BrokenPipeError, ValueError, OSError):
                continue
