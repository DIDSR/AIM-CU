AIM-CU: A CUSUM-based tool for AI Monitoring
======

.. image:: assets/AIM-CU-Overview.png
    :width: 800
    :align: center
    :alt: AIM-CU flowchart

Monitoring a clinically deployed AI device to detect performance drift is an essential step to ensure the safety and effectiveness of AI. 

AIM-CU is a statistical tool for AI monitoring based on a cumulative sum (AIM-CU) approach.

AIM-CU computes:

* The parameter choices for change-point detection based on an acceptable false alarm rate
* Detection delay estimates for a given displacement of the performance metric from the target for those parameter choices.

System setup
------------
Make sure R is installed in the system. There is no specific version that this relies on, but here we have used R version 4.1.2 (2021-11-01). 
Instructions for Linux (the below setup is only performed in Linux):

.. code-block:: shell

    wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc |  tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
    add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
    apt-get install -y --no-install-recommends r-base r-base-dev

    # setup R configs
    echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.us.r-project.org'; options(repos = r);" > ~/.Rprofile
    Rscript -e "install.packages('ggplot2')"
    Rscript -e "install.packages('hexbin')"
    Rscript -e "install.packages('lazyeval')"
    Rscript -e "install.packages('cusumcharter')"
    Rscript -e "install.packages('RcppCNPy')"
    Rscript -e "install.packages('spc')"

Code execution
--------------
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

    cd src/package
    python3 app.py

Open the URL http://0.0.0.0:7860 that is running the AIM-CU locally.

Example code execution
----------------------
Example code can be run through Jupyter Notebook. Do this by entering the ``jupyter notebook`` command from the ``/src/package/`` directory. The tool is designed to be used through a GUI, not from the console.

Demo
----
AIM-CU can also be run through the demo available at https://huggingface.co/spaces/didsr/AIM-CU. If Space is paused, click on Restart button.

Usability
---------
* Example AI output CSV file is available as `config/spec-60-60.csv <config/spec-60-60.csv>`_ to be uploaded in monitoring phase.

* Workflow instruction to run the tool is available at bottom-left of UI.

* Sample UI output is available at `assets/ui.png <assets/ui.png>`_.

* Setting ``control:save_figure`` to ``true`` from `config.toml <config/config.toml>`_ will save tables and plots in `figure/ <figure/>`_.

* Running AIM-CU usually only takes a few seconds, and it does not require a GPU to run.

Related References
------------------
* Smriti Prathapan, Ravi K. Samala, Nathan Hadjiyski, Pierre‑François D’Haese, Nicholas Petrick, Jana Delfino, Fabien Maldonado, Brandon Nelson, Ghada Zamzmi, Phuong Nguyen, Yelena Yesha, and Berkman Sahiner, "Detecting performance drift in AI models for medical image analysis using CUSUM chart" (Journal Draft in-progress 2025)

* Prathapan, S., Sahiner, B.,  Kadia, D., and Samala, R.K. 2025, AIM-CU: A statistical tool for AI Monitoring. In Medical Imaging 2025: Computer-Aided Diagnosis. SPIE. (Accepted)

* Prathapan, S., Samala, R.K., Hadjiyski, N., D’Haese, P.F., Maldonado, F., Nguyen, P., Yesha, Y. and Sahiner, B., 2024, April. Quantifying input data drift in medical machine learning models by detecting change-points in time-series data. In Medical Imaging 2024: Computer-Aided Diagnosis (Vol. 12927, pp. 67-76). SPIE. https://doi.org/10.1117/12.3008771

* Smriti Prathapan, Ravi K. Samala, Nathan Hadjiyski, Pierre‑François D’Haese, Nicholas Petrick, Jana Delfino, Fabien Maldonado, Brandon Nelson, Ghada Zamzmi, Phuong Nguyen, Yelena Yesha, and Berkman Sahiner, "Post-market Monitoring of AI-enabled Medical Devices for Radiology and Healthcare Applications" (FDA-UMiami Collaboration Poster, September 2023)

Disclaimer
----------
This software and documentation was developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.
