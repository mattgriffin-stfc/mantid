.. algorithm::

.. summary::

.. relatedalgorithms::

.. properties::

Description
-----------

This algorithm removes the background present in PSI bin data, loaded using LoadPSIMuonBin

The background is removed from the original data through the following formula,

.. math:: y_{cor}(t) = y_{org}(t) - B,

where :math:`y_{cor}(t)` is the corrected bin data, :math:`y_{org}(t)` the original counts data and :math:`B` is the
flat background.

To obtain the flat background, :math:`B`, the second-half of the raw-data is fitted with the following function,

.. math:: f(t) = \mbox{A}e^{-\lambda t} + B

where the first term represents a ExpDecay function, see :ref:`func-ExpDecayMuon`.

The algorithm takes in an input workspace and performs the correction inplace on the workspace. The number of iterations
can be specified through an optional parameter. If a poor quality fit is returned, a warning will be displayed in the
Mantid logger.

Usage
-----

.. testcode:: CalculateBackgroundForTestData

    # import mantid algorithms, numpy and matplotlib
    from mantid.simpleapi import *
    import numpy as np
    # Generate shifted ExpDecay data
    A = 1200
    Background = 20
    Lambda = 0.5;
    time = np.linspace(0, 10, 100)
    func = lambda t: A*np.exp(-Lambda*t) + Background
    counts = np.array([func(ti) for ti in time])

    # Create workspaces
    input_workspace = CreateWorkspace(time, counts)
    input_workspace.setYUnit("Counts")
    workspace_copy = input_workspace.clone()

    # Run PSIBackgroundSubtraction Algorithm
    PSIBackgroundSubtraction(input_workspace)

    # Find the difference between the workspaces
    workspace_diff = Minus(workspace_copy, input_workspace)
    # The counts in workspace diff should be a flat line corresponding to the background
    print("Differences in counts are: {}".format(workspace_diff.dataY(0)))

Output:

.. testoutput:: CalculateBackgroundForTestData

    Differences in counts are: [ 20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.
      20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.
      20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.
      20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.
      20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.
      20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.  20.
      20.  20.  20.  20.  20.  20.  20.  20.  20.  20.]

.. categories::

.. sourcelink::
    :h: Framework/Muon/inc/MantidMuon/PSIBackgroundSubtraction.h
    :cpp: Framework/Muon/src/PSIBackgroundSubtraction.cpp
