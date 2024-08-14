FROM ubuntu:22.04

# RUN apt-get update && apt-get install -y python3 python3-pip

# #
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common dirmngr wget
# RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get install -y python3.8 python3.8-distutils python3.8-dev
# RUN wget https://bootstrap.pypa.io/get-pip.py && python3.8 get-pip.py && python3.8 -m pip install --upgrade pip

RUN apt-get update && apt-get install -y python3 python3-distutils python3-dev
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py

RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc |  tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
RUN apt-get install -y --no-install-recommends r-base r-base-dev

# # --no-cache-dir --upgrade

# setup R configs
RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.us.r-project.org'; options(repos = r);" > ~/.Rprofile
RUN Rscript -e "install.packages('ggplot2')"
RUN Rscript -e "install.packages('hexbin')"
RUN Rscript -e "install.packages('lazyeval')"
RUN Rscript -e "install.packages('cusumcharter')"
RUN Rscript -e "install.packages('RcppCNPy')"
RUN Rscript -e "install.packages('spc')"
#

# #new
# # Set up a new user named "user" with user ID 1000
# RUN useradd -m -u 1000 user

# # Switch to the "user" user
# USER user

# # Set home to the user's home directory
# ENV HOME=/home/user \
#     PATH=/home/user/.local/bin:$PATH

# # Set the working directory to the user's home directory
# WORKDIR $HOME/app

# # Copy the current directory contents into the container at $HOME/app setting the owner to the user
# COPY --chown=user . $HOME/app
# RUN pip install --no-cache-dir --upgrade -r requirements.txt
# #new

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN pip install --no-cache-dir --upgrade rpy2

CMD ["python3", "ARLTheoretical.py"]