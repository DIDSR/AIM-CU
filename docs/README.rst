Documentation
=============

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