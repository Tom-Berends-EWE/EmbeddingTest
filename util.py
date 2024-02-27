__all__ = ['create_psql_process']

import os
from subprocess import Popen


def create_psql_process() -> Popen:
    return Popen([os.environ['PSQL_SERVER_SETUP_CMD']])
