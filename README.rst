AIM-CU
======
A CUSUM-based tool for AI Monitoring

AIM-CU is a statistical tool for AI monitoring using cumulative sum (AIM-CU). AIM-CU computes:

* The parameter choices for change-point detection based on an acceptable false alarm rate
* Detection delay estimates for a given displacement of the performance metric from the target for those parameter choices.

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

    python3 app.py

Open the URL http://0.0.0.0:7860 that is running the AIM-CU locally.

Demo
----
AIM-CU can also be run through the demo available at https://huggingface.co/spaces/didsr/AIM-CU. If Space is paused, click on Restart button.

Related References
------------------

* Prathapan, S., Sahiner, B.,  Kadia, D., and Samala, R.K. 2025, AIM-CU: A statistical tool for AI Monitoring. In Medical Imaging 2025: Computer-Aided Diagnosis. SPIE. (Accepted)

* Prathapan, S., Samala, R.K., Hadjiyski, N., D’Haese, P.F., Maldonado, F., Nguyen, P., Yesha, Y. and Sahiner, B., 2024, April. Quantifying input data drift in medical machine learning models by detecting change-points in time-series data. In Medical Imaging 2024: Computer-Aided Diagnosis (Vol. 12927, pp. 67-76). SPIE. https://doi.org/10.1117/12.3008771

* Smriti Prathapan, Ravi K. Samala, Nathan Hadjiyski, Pierre‑François D’Haese, Nicholas Petrick, Jana Delfino, Fabien Maldonado, Brandon Nelson, Ghada Zamzmi, Phuong Nguyen, Yelena Yesha, and Berkman Sahiner, "Post-market Monitoring of AI-enabled Medical Devices for Radiology and Healthcare Applications" (FDA-UMiami Collaboration Poster, September 2023)


Disclaimer
----------
This software and documentation was developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.
