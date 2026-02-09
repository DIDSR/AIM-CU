#!/bin/bash
# Enable permission first: chmod +x setup.sh

ENV_NAME="aimcu1"

eval "$(conda shell.bash hook)"

# Create a conda environment
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

# Install rpy2 from conda-forge
conda install -c conda-forge rpy2=3.5.1 -y

# setup R configs
echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.us.r-project.org'; options(repos = r);" > ~/.Rprofile
Rscript -e "install.packages('ggplot2')"
Rscript -e "install.packages('hexbin')"
Rscript -e "install.packages('lazyeval')"
Rscript -e "install.packages('cusumcharter')"
Rscript -e "install.packages('RcppCNPy')"
Rscript -e "install.packages('spc')"

# Install everything but rpy2 in pip (rpy2 is commented)
pip install -r requirements_conda.txt

# Latest version of below packages may be required
pip install --upgrade gradio huggingface_hub

bash run.sh