#!/bin/bash

# Required everytime you open the new terminal
export http_proxy=; export https_proxy=

USER="dhaval.kadia"
ENV_NAME="aimcu11"

source /home/"$USER"/miniconda3/bin/activate /home/"$USER"/miniconda3/envs/"$ENV_NAME"

cd src/package
python3 app.py