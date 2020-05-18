=========================
Indirect Geometry Changes
=========================

.. contents:: Table of Contents
   :local:

Improvements
############

- An issue with IndirectDataReduction causing it to slow down processing when open has been resolved.
- Scientific constraints have been added to all fitting tabs in IndirectDataAnalysis.
- The Abins python library in *scripts* has been substantially
  re-organised, including a re-name from ``AbinsModules`` to
  ``abins``. These changes should make the library more approachable and maintainable.
  They should not impact functionality of the Abins Algorithm, but will break any user python scripts
  that import ``AbinsModules``.
- :ref:`CalculateMonteCarloAbsorption <algm-CalculateMonteCarloAbsorption>` will now work also for ILL fixed window scan reduced data, in which case the correction will be calculated for elastic wavelength.


Bug Fixes
#########

- FQ and Msd tabs now label output workspaces with the fitting function.
- :ref:`CalculateMonteCarloAbsorption <algm-CalculateMonteCarloAbsorption>` will now work correctly for annular sample in a container.

:ref:`Release 5.1.0 <v5.1.0>`
