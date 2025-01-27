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

.. code-block:: shell

    sphinx-build -M simplepdf ../../docs/source ../../docs/build --fail-on-warning

Navigate to ``/docs/build`` and find ``html`` and ``simplepdf`` folders.

.. sphinx-build -b rinoh ../../docs/source ../../docs/build --fail-on-warning