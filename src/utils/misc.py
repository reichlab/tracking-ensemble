"""
Utilities for working with distributions and similar stuff
"""

import os
import os.path as path


def ensure_dir(directory: str) -> str:
    if not path.exists(directory):
        os.makedirs(directory)
    return directory
