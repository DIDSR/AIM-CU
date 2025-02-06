.. AIM-CU documentation master file, created by
   sphinx-quickstart on Wed Nov 27 09:59:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AIM-CU documentation
====================

A CUSUM-based tool for AI Monitoring

AIM-CU is a statistical tool for AI monitoring using cumulative sum (AIM-CU). AIM-CU computes:

* The parameter choices for change-point detection based on an acceptable false alarm rate
* Detection delay estimates for a given displacement of the performance metric from the target for those parameter choices.

System setup
------------
Make sure R is installed in the system. Instructions for linux:

.. code-block:: shell

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
    
Code execution
------------------
Clone AIM-CU repository.

.. code-block:: shell

   git clone https://github.com/DIDSR/AIM-CU.git

Run the following commands to install required dependencies (Python = 3.10 is used).

.. code-block:: shell

   apt-get -y install python3
   apt-get -y install pip
   cd AIM-CU
   pip install -r requirements.txt

Run AIM-CU.

.. code-block:: shell

   python3 app.py

Open the URL http://0.0.0.0:7860 that is running the AIM-CU locally.

Demo
------------------
AIM-CU can also be run through the demo available at https://huggingface.co/spaces/didsr/AIM-CU. If Space is paused, click on Restart button.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ref_method
   ref_cusum
   ref_theoretical
   ref_utils