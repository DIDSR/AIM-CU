Methods
=======

CUSUM parameters
----------------

.. csv-table:: CUSUM parameters
   :file: ../../assets/params.csv
   :header-rows: 1

CUSUM chart
-----------

A two-sided CUSUM control chart computes the cumulative differences or
deviations of individual observations from the target mean (or
in-control mean, :math:`\mu_{in}`). The positive and negative cumulative
sums are calculated:

.. math::

   \\ S_{hi}(d) = max(0, S_{hi}(d-1) + x_d  - \hat{\mu}_{in} - K)
   \\ S_{lo}(d) = max(0, S_{lo}(d-1) - x_d +  \hat{\mu}_{in} - K)

where *d* denotes a unit of time, :math:`x_d` is the value of quantity
being monitored at time :math:`d`, :math:`\hat{\mu}_{in}` is the
in-control mean of :math:`x_d`, and :math:`K` is a "reference value"
related to the magnitude of change that one is interested in detecting.
:math:`S_{hi}` and :math:`S_{lo}` are the cumulative sum of positive and
negative changes. To detect a change in the observed values from the
in-control mean, the CUSUM scheme accumulates deviations that are
:math:`K` units away from the in-control mean. Let :math:`\sigma_{in}`
denote the in-control standard deviation of :math:`x_d`.