#!/bin/bash
set -e

if [ $# -eq 0 ]; then
   exec jupyter notebook --notebook-dir=/notebooks --ip='0.0.0.0' --port=8888 --no-browser --allow-root
fi


if [ "$1" = 'jupyterlab' ]; then
    exec jupyter lab --notebook-dir=/notebooks --ip='0.0.0.0' --port=8888 --no-browser --allow-root
elif [ "$1" = 'jupyter' ]; then
   exec jupyter notebook --notebook-dir=/notebooks --ip='0.0.0.0' --port=8888 --no-browser --allow-root
else
    exec "$@"
fi
