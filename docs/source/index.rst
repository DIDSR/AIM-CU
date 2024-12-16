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
AIM-CU can also be run through its demo available at https://huggingface.co/didsr/AIM-CU.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ref_cusum
   ref_theoretical
   ref_utils

