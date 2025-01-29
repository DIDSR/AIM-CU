Documentation
=============

Read the documentation
----------------------
Access the documentation from `index.html <build/html/index.html>`_

Generate Sphinx documentation
-----------------------------
Run the below command from ``/src/package``.

HTML format:

.. code-block:: shell

    sphinx-build -M html ../../docs/source ../../docs/build --fail-on-warning

As a PDF file:

Required: Install latex - ``sudo apt-get install texlive-full``

.. code-block:: shell

    sphinx-build -M latexpdf ../../docs/source ../../docs/build --fail-on-warning

Access it from `aim-cu.pdf <build/latex/aim-cu.pdf>`_