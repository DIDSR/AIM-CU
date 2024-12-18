FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common dirmngr wget

RUN apt-get update && apt-get install -y python3 python3-distutils python3-dev
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py

RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc |  tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
RUN apt-get install -y --no-install-recommends r-base r-base-dev

# setup R configs
RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.us.r-project.org'; options(repos = r);" > ~/.Rprofile
RUN Rscript -e "install.packages('ggplot2')"
RUN Rscript -e "install.packages('hexbin')"
RUN Rscript -e "install.packages('lazyeval')"
RUN Rscript -e "install.packages('cusumcharter')"
RUN Rscript -e "install.packages('RcppCNPy')"
RUN Rscript -e "install.packages('spc')"

WORKDIR /app/src/package
COPY . /app/

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

CMD ["python3", "app.py"]