.. algorithm::

.. summary::

.. relatedalgorithms::

.. properties::

Description
-----------

This algorithm performs a Monte Carlo simulation to calculate the correction factors due
to attenuation & single scattering within a sample plus optionally its sample environment.

This adaptation of the :ref:`MonteCarloAbsorption <algm-MonteCarloAbsorption>` exploits the 
high throughput attribute of a GPU to generation corrections faster whilst maintaining 
consistency and usability.


CUDA Acceleration
############################

This algorithm utilises CUDA, executing the main body of work on the GPU. Parallelism occurs at
the spectra, bin and event level which allows for the highest possible utilization and performance.

A CUDA enabled GPU must match the following prerequisites to execute this algorithm:

- Support Compute Capability 6 (Pascal) or later
- Latest device drivers (or drivers which support CUDA Toolkit/Runtime 7.0 or later)

If your device does not meet these requirements then the algorithm will output an error in the 
log window.

Limitations
############################

This CUDA implementation lacks many of the :ref:`MonteCarloAbsorption <algm-MonteCarloAbsorption>` 
algorithms core features, imposing the following restrictions:

- Sample and Environment objects must be defined using constructive solid geometry (CSGObject).  
- Sparse Instruments and other interpolation types are not supported.
- Cannot reuse tracks for different wavelengths (generates a new track for every wavelength).
- Strategy does not generate a scatter report.
- Input and output workspace size must not exceed the total available device memory.
- Input workspace must not exceed :math:`2^32 - 1` spectra and :math:`65535` bins.
- EventsPerPoint must not exceed :math:`65535`.

Input Workspace Requirements
############################

The algorithm will compute correction factors on a bin-by-bin basis for each spectrum within
the input workspace. The following assumptions on the input workspace are made:

- X units are in wavelength
- the instrument is fully defined
- properties of the sample and optionally its environment have been set with
  :ref:`SetSample <algm-SetSample>`

By default the beam is assumed to be the a slit with width and height matching
the width and height of the sample. This can be overridden using :ref:`SetBeam <algm-SetBeam>`.

.. note:: The input must not contain varying bin widths, as interpolation is not currently supported.

Method
######

By default, the material for the sample & containers will define the values of the cross section used to compute the absorption factor and will
include contributions from both the total scattering cross section & absorption cross section.
This follows the Hamilton-Darwin [#DAR]_, [#HAM]_ approach as described by T. M. Sabine in the International Tables of Crystallography Vol. C [#SAB]_.

The algorithm proceeds as follows. In parallel for each of the `NEvents` in a bin within a spectrum:

#. find the associated detector position

#. find the associated efixed value (if applicable) & convert to wavelength (:math:`\lambda_{fixed}`)

#. generate a random point on the beam face defined by the input height & width. If the point is outside of the
   area defined by the face of the sample then it is pulled to the boundary of this area

#. generate a random point within the sample or container objects as the scatter point and create a `Track`
   from the selected position on the beam face to the scatter point

#. test for intersections of the track & sample/container objects, giving the number of subsections
   and corresponding distances within the object for each section, call them :math:`l_{1i}`

#. form a second `Track` with the scatter position as the starting point and the direction defined by
   `detPos - scatterPos`

#. test for intersections of the track & sample/container objects, giving the number of subsections
   and corresponding distances within the object for each section, call them :math:`l_{2i}`


#. define :math:`\lambda_1` as the wavelength before scattering & :math:`\lambda_2` as wavelength after scattering:

   - Direct: :math:`\lambda_1 = \lambda_1`, :math:`\lambda_2 = \lambda_{step}`

   - Indirect: :math:`\lambda_1 = \lambda_{step}`, :math:`\lambda_2 = \lambda_{fixed}`

   - Elastic: :math:`\lambda_1 = \lambda_2 = \lambda_{step}`

#. compute the self-attenuation factor for all intersections as
   :math:`\prod\limits_{i} \exp(-(\rho_{1i}\sigma_{1i}(\lambda_{1i})l_{1i} + \rho_{2i}\sigma_{2i}(\lambda_{2i})l_{2i}))`
   where :math:`\rho` is the mass density of the material &
   :math:`\sigma` the absorption cross-section at a given wavelength

#. accumulate this wavelength-specific factor across all `NEvents`

#. average the accumulated attentuation factors over `NEvents` and assign this as the correction factor for
   this :math:`\lambda_{step}`.


Usage
-----

**Example: A cylindrical sample with no container**

.. testcode:: ExCylinderSampleOnly

   data = CreateSampleWorkspace(WorkspaceType='Histogram', NumBanks=1)
   data = ConvertUnits(data, Target="Wavelength")
   # Default up axis is Y
   SetSample(data, Geometry={'Shape': 'Cylinder', 'Height': 5.0, 'Radius': 1.0,
                     'Center': [0.0,0.0,0.0]},
                   Material={'ChemicalFormula': '(Li7)2-C-H4-N-Cl6', 'SampleNumberDensity': 0.07})

   abscor = CudaMonteCarloAbsorption(data)
   corrected = data/abscor

**Example: A cylindrical sample setting a beam size**

.. testcode:: ExCylinderSampleAndBeamSize

   data = CreateSampleWorkspace(WorkspaceType='Histogram', NumBanks=1)
   data = ConvertUnits(data, Target="Wavelength")
   # Default up axis is Y
   SetSample(data, Geometry={'Shape': 'Cylinder', 'Height': 5.0, 'Radius': 1.0,
                     'Center': [0.0,0.0,0.0]},
                     Material={'ChemicalFormula': '(Li7)2-C-H4-N-Cl6', 'SampleNumberDensity': 0.07})
   SetBeam(data, Geometry={'Shape': 'Slit', 'Width': 0.8, 'Height': 1.0})

   abscor = MonteCarloAbsorption(data, NumberOfWavelengthPoints=50)
   corrected = data/abscor

**Example: A cylindrical sample with predefined container**

The following example uses a test sample environment defined for the ``TEST_LIVE``
facility and ``ISIS_Histogram`` instrument and assumes that these are set as the
default facility and instrument respectively. The definition can be found at
``[INSTALLDIR]/instrument/sampleenvironments/TEST_LIVE/ISIS_Histogram/CRYO-01.xml``.

.. testsetup:: ExCylinderPlusEnvironment

   FACILITY_AT_START = config['default.facility']
   INSTRUMENT_AT_START = config['default.instrument']
   config['default.facility'] = 'TEST_LIVE'
   config['default.instrument'] = 'ISIS_Histogram'

.. testcleanup:: ExCylinderPlusEnvironment

   config['default.facility'] = FACILITY_AT_START
   config['default.instrument'] = INSTRUMENT_AT_START

.. testcode:: ExCylinderPlusEnvironment

   data = CreateSampleWorkspace(WorkspaceType='Histogram', NumBanks=1)
   data = ConvertUnits(data, Target="Wavelength")
   # Sample geometry is defined by container but not completely filled so
   # we just define the height
   SetSample(data, Environment={'Name': 'CRYO-01', 'Container': '8mm'},
             Geometry={'Height': 4.0},
             Material={'ChemicalFormula': '(Li7)2-C-H4-N-Cl6', 'SampleNumberDensity': 0.07})

   abscor = MonteCarloAbsorption(data)
   corrected = data/abscor

References
----------

.. [#DAR] Darwin, C. G., *Philos. Mag.*, **43** 800 (1922)
          `doi: 10.1080/10448639208218770 <http://dx.doi.org/10.1080/10448639208218770>`_
.. [#HAM] Hamilton, W.C., *Acta Cryst*, **10**, 629 (1957)
          `doi: 10.1107/S0365110X57002212 <http://dx.doi.org/10.1107/S0365110X57002212>`_
.. [#SAB] Sabine, T. M., *International Tables for Crystallography*, Vol. C, Page 609, Ed. Wilson, A. J. C and Prince, E. Kluwer Publishers (2004)
          `doi: 10.1107/97809553602060000103 <http://dx.doi.org/10.1107/97809553602060000103>`_

|

.. categories::

.. sourcelink::
