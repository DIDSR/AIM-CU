Methods
=======

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

.. _subsec3:

Performance Assessment for CUSUM
--------------------------------

Let *d=0,...n-1* indicate the days (or time), where *n* is the total
length of time during which the AI model performance is monitored. We
assume that the AI model performance is in-control between days 0 and
:math:`d_s`, :math:`d_s<d_{n-1}`, and a change in performance occurs on
day :math:`d_s` such that :math:`x_{0...d_{s-1}}` and
:math:`x_{d_{s}...d_{n-1}}` are AI performance measure values for the
pre-change and the post-change periods respectively.

In this setting, three different cases are of interest:

-  Change-point is detected at :math:`d_d : d_d \ge d_s` with a
   detection delay :math:`d_d - d_s`.

-  Change-point is detected at :math:`d_d : d_d < d_s`. This is called a
   false alarm (type-I error).

-  Change-point is not detected. This is a missed detection (type-II
   error).

Mean time between false alarms (MTBFA)
--------------------------------------

.. math::

   \label{MTBFA_estimate}
       \widehat{MTBFA} = \frac{\sum_{j=1} ^{N} z_j}{\sum_{j=1} ^{N} d_j}

where :math:`N` is the number of independent experiments, :math:`d_j` is
a binary value for each experiment indicating whether or not a change
was detected in the pre-change regime, :math:`d_j \in \{0,1\}` and
:math:`z_j = min(d_d(j), d_s)`.

Average Detection Delay (ADD)
-----------------------------

.. math::

   \label{ADD_estimate}
       \widehat{ADD} = \frac{\sum_{j=1} ^{N} z_j- d_s}{\sum_{j=1} ^{N} c_j}

where :math:`c_j` is a binary value for each experiment indicating
whether or not a change was detected in the post change regime,
:math:`c_j \in \{0,1\}` and :math:`z_j = min(d_d(j), n)`.
