#!/bin/bash
set -ex -o pipefail

# use exec to replace the shell and ensure that SIGINT is sent to the app
exec python -m app
