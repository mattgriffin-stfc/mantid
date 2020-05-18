# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
from mantid.api import *
from mantid.kernel import *
from mantid.simpleapi import *
from os import path


EMPTY_TOKEN = '000000'


def needs_loading(property_value, loading_reduction_type):
    """
        Checks whether a given unary input needs loading or is already loaded in ADS.
        @param property_value : the string value of the corresponding FileProperty
        @param loading_reduction_type : the reduction_type of input to load
    """
    loading = False
    ws_name = ''
    if property_value:
        ws_name = path.splitext(path.basename(property_value))[0]
        if mtd.doesExist(ws_name):
            logger.notice('Reusing {0} workspace: {1}'.format(loading_reduction_type, ws_name))
        else:
            loading = True
    return [loading, ws_name]


def get_run_number(value):
    """
        Extracts the run number from the first run out of the string value of a multiple file property of numors
    """
    return path.splitext(path.basename(value.split(',')[0].split('+')[0]))[0]


def needs_processing(property_value, process_reduction_type):
    """
        Checks whether a given unary reduction needs processing or is already cached in ADS with expected name.
        @param property_value : the string value of the corresponding MultipleFile input property
        @param process_reduction_type : the reduction_type of process
    """
    do_process = False
    ws_name = ''
    if property_value:
        run_number = get_run_number(property_value)
        ws_name = run_number + '_' + process_reduction_type
        if mtd.doesExist(ws_name):
            run = mtd[ws_name].getRun()
            if run.hasProperty('ProcessedAs'):
                process = run.getLogData('ProcessedAs').value
                if process == process_reduction_type:
                    logger.notice('Reusing {0} workspace: {1}'.format(process_reduction_type, ws_name))
                else:
                    logger.warning(
                        '{0} workspace found, but processed differently: {1}'.format(process_reduction_type, ws_name))
                    do_process = True
            else:
                logger.warning(
                    '{0} workspace found, but missing the ProcessedAs flag: {1}'.format(process_reduction_type, ws_name))
                do_process = True
        else:
            do_process = True
    return [do_process, ws_name]


class SANSILLAutoProcess(DataProcessorAlgorithm):
    """
    Performs complete treatment of ILL SANS data; instruments D11, D16, D22, D33.
    """
    progress = None
    reduction_type = None
    sample = None
    absorber = None
    beam = None
    container = None
    stransmission = None
    ctransmission = None
    btransmission = None
    atransmission = None
    sensitivity = None
    mask = None
    flux = None
    default_mask = None
    output = None
    output_sens = None
    dimensionality = None
    reference = None
    normalise = None
    radius = None
    thickness = None
    theta_dependent = None

    def category(self):
        return 'ILL\\SANS;ILL\\Auto'

    def summary(self):
        return 'Performs complete SANS data reduction at the ILL.'

    def seeAlso(self):
        return ['SANSILLReduction', 'SANSILLIntegration',]

    def name(self):
        return 'SANSILLAutoProcess'

    def validateInputs(self):
        result = dict()
        message = 'Wrong number of {0} runs: {1}. Provide one or as many as sample runs: {2}.'
        tr_message = 'Wrong number of {0} runs: {1}. Provide one or multiple runs summed with +.'
        sample_dim = self.getPropertyValue('SampleRuns').count(',')
        abs_dim = self.getPropertyValue('AbsorberRuns').count(',')
        beam_dim = self.getPropertyValue('BeamRuns').count(',')
        flux_dim = self.getPropertyValue('FluxRuns').count(',')
        can_dim = self.getPropertyValue('ContainerRuns').count(',')
        str_dim = self.getPropertyValue('SampleTransmissionRuns').count(',')
        ctr_dim = self.getPropertyValue('ContainerTransmissionRuns').count(',')
        btr_dim = self.getPropertyValue('TransmissionBeamRuns').count(',')
        atr_dim = self.getPropertyValue('TransmissionAbsorberRuns').count(',')
        mask_dim = self.getPropertyValue('MaskFiles').count(',')
        sens_dim = self.getPropertyValue('SensitivityMaps').count(',')
        ref_dim = self.getPropertyValue('ReferenceFiles').count(',')
        if self.getPropertyValue('SampleRuns') == '':
            result['SampleRuns'] = 'Please provide at least one sample run.'
        if abs_dim != sample_dim and abs_dim != 0:
            result['AbsorberRuns'] = message.format('Absorber', abs_dim, sample_dim)
        if beam_dim != sample_dim and beam_dim != 0:
            result['BeamRuns'] = message.format('Beam', beam_dim, sample_dim)
        if can_dim != sample_dim and can_dim != 0:
            result['ContainerRuns'] = message.format('Container', can_dim, sample_dim)
        if str_dim != 0:
            result['SampleTransmissionRuns'] = tr_message.format('SampleTransmission', str_dim)
        if ctr_dim != 0:
            result['ContainerTransmissionRuns'] = tr_message.format('ContainerTransmission', ctr_dim)
        if btr_dim != 0:
            result['TransmissionBeamRuns'] = tr_message.format('TransmissionBeam', btr_dim)
        if atr_dim != 0:
            result['TransmissionAbsorberRuns'] = tr_message.format('TransmissionAbsorber', atr_dim)
        if mask_dim != sample_dim and mask_dim != 0:
            result['MaskFiles'] = message.format('Mask', mask_dim, sample_dim)
        if ref_dim != sample_dim and ref_dim != 0:
            result['ReferenceFiles'] = message.format('Reference', ref_dim, sample_dim)
        if sens_dim != sample_dim and sens_dim != 0:
            result['SensitivityMaps'] = message.format('Sensitivity', sens_dim, sample_dim)
        if flux_dim != flux_dim and flux_dim != 0:
            result['FluxRuns'] = message.format('Flux')

        return result

    def setUp(self):
        self.sample = self.getPropertyValue('SampleRuns').split(',')
        self.absorber = self.getPropertyValue('AbsorberRuns').split(',')
        self.beam = self.getPropertyValue('BeamRuns').split(',')
        self.flux = self.getPropertyValue('FluxRuns').split(',')
        self.container = self.getPropertyValue('ContainerRuns').split(',')
        self.stransmission = self.getPropertyValue('SampleTransmissionRuns')
        self.ctransmission = self.getPropertyValue('ContainerTransmissionRuns')
        self.btransmission = self.getPropertyValue('TransmissionBeamRuns')
        self.atransmission = self.getPropertyValue('TransmissionAbsorberRuns')
        self.sensitivity = self.getPropertyValue('SensitivityMaps').split(',')
        self.default_mask = self.getPropertyValue('DefaultMaskFile')
        self.mask = self.getPropertyValue('MaskFiles').split(',')
        self.reference = self.getPropertyValue('ReferenceFiles').split(',')
        self.output = self.getPropertyValue('OutputWorkspace')
        self.output_sens = self.getPropertyValue('SensitivityOutputWorkspace')
        self.normalise = self.getPropertyValue('NormaliseBy')
        self.theta_dependent = self.getProperty('ThetaDependent').value
        self.radius = self.getProperty('BeamRadius').value
        self.dimensionality = len(self.sample)
        self.progress = Progress(self, start=0.0, end=1.0, nreports=10 * self.dimensionality)

    def PyInit(self):

        self.declareProperty(WorkspaceGroupProperty('OutputWorkspace', '',
                                                    direction=Direction.Output),
                             doc='The output workspace group containing reduced data.')

        self.declareProperty(MultipleFileProperty('SampleRuns',
                                                  action=FileAction.OptionalLoad,
                                                  extensions=['nxs'],
                                                  allow_empty=True),
                             doc='Sample run(s).')

        self.declareProperty(MultipleFileProperty('AbsorberRuns',
                                                  action=FileAction.OptionalLoad,
                                                  extensions=['nxs']),
                             doc='Absorber (Cd/B4C) run(s).')

        self.declareProperty(MultipleFileProperty('BeamRuns',
                                                  action=FileAction.OptionalLoad,
                                                  extensions=['nxs']),
                             doc='Empty beam run(s).')

        self.declareProperty(MultipleFileProperty('FluxRuns',
                                                  action=FileAction.OptionalLoad,
                                                  extensions=['nxs']),
                             doc='Empty beam run(s) for flux calculation only; '
                                 'if left blank flux will be calculated from BeamRuns.')

        self.declareProperty(MultipleFileProperty('ContainerRuns',
                                                  action=FileAction.OptionalLoad,
                                                  extensions=['nxs']),
                             doc='Empty container run(s).')

        self.setPropertyGroup('SampleRuns', 'Numors')
        self.setPropertyGroup('AbsorberRuns', 'Numors')
        self.setPropertyGroup('BeamRuns', 'Numors')
        self.setPropertyGroup('FluxRuns', 'Numors')
        self.setPropertyGroup('ContainerRuns', 'Numors')

        self.declareProperty(MultipleFileProperty('SampleTransmissionRuns',
                                                  action=FileAction.OptionalLoad,
                                                  extensions=['nxs']),
                             doc='Sample transmission run(s).')

        self.declareProperty(MultipleFileProperty('ContainerTransmissionRuns',
                                                  action=FileAction.OptionalLoad,
                                                  extensions=['nxs']),
                             doc='Container transmission run(s).')

        self.declareProperty(MultipleFileProperty('TransmissionBeamRuns',
                                                  action=FileAction.OptionalLoad,
                                                  extensions=['nxs']),
                             doc='Empty beam run(s) for transmission.')

        self.declareProperty(MultipleFileProperty('TransmissionAbsorberRuns',
                                                  action=FileAction.OptionalLoad,
                                                  extensions=['nxs']),
                             doc='Absorber (Cd/B4C) run(s) for transmission.')

        self.setPropertyGroup('SampleTransmissionRuns', 'Transmissions')
        self.setPropertyGroup('ContainerTransmissionRuns', 'Transmissions')
        self.setPropertyGroup('TransmissionBeamRuns', 'Transmissions')
        self.setPropertyGroup('TransmissionAbsorberRuns', 'Transmissions')
        self.copyProperties('SANSILLReduction',
                            ['ThetaDependent'])
        self.setPropertyGroup('ThetaDependent', 'Transmissions')

        self.declareProperty(MultipleFileProperty('SensitivityMaps',
                                                  action=FileAction.OptionalLoad,
                                                  extensions=['nxs']),
                             doc='File(s) containing the map of relative detector efficiencies.')

        self.declareProperty(FileProperty('DefaultMaskFile', '', action=FileAction.OptionalLoad, extensions=['nxs']),
                             doc='File containing the default mask to be applied to all the detector configurations.')

        self.declareProperty(MultipleFileProperty('MaskFiles',
                                                  action=FileAction.OptionalLoad,
                                                  extensions=['nxs']),
                             doc='File(s) containing the beam stop and other detector mask.')

        self.declareProperty(MultipleFileProperty('ReferenceFiles',
                                                  action=FileAction.OptionalLoad,
                                                  extensions=['nxs']),
                             doc='File(s) containing the corrected water data for absolute normalisation.')

        self.declareProperty(MatrixWorkspaceProperty('SensitivityOutputWorkspace', '',
                                                     direction=Direction.Output,
                                                     optional=PropertyMode.Optional),
                             doc='The output sensitivity map workspace.')

        self.copyProperties('SANSILLReduction', ['NormaliseBy'])

        self.declareProperty('SampleThickness', 0.1, validator=FloatBoundedValidator(lower=0.),
                             doc='Sample thickness [cm]')

        self.declareProperty('BeamRadius', 0.05, validator=FloatBoundedValidator(lower=0.),
                             doc='Beam radius [m]; used for beam center finding, transmission and flux calculations.')

        self.declareProperty('WaterCrossSection', 1., doc='Provide water cross-section; '
                                                          'used only if the absolute scale is done by dividing to water.')

        self.setPropertyGroup('SensitivityMaps', 'Options')
        self.setPropertyGroup('DefaultMaskFile', 'Options')
        self.setPropertyGroup('MaskFiles', 'Options')
        self.setPropertyGroup('ReferenceFiles', 'Options')
        self.setPropertyGroup('SensitivityOutputWorkspace', 'Options')
        self.setPropertyGroup('NormaliseBy', 'Options')
        self.setPropertyGroup('SampleThickness', 'Options')
        self.setPropertyGroup('BeamRadius', 'Options')
        self.setPropertyGroup('WaterCrossSection', 'Options')

        self.copyProperties('SANSILLIntegration',
                            ['OutputType', 'CalculateResolution', 'DefaultQBinning', 'BinningFactor', 'OutputBinning',
                             'NPixelDivision', 'NumberOfWedges', 'WedgeWorkspace', 'WedgeAngle', 'WedgeOffset',
                             'AsymmetricWedges', 'MaxQxy', 'DeltaQ', 'IQxQyLogBinning', 'PanelOutputWorkspaces'])

        self.setPropertyGroup('OutputType', 'Integration Options')
        self.setPropertyGroup('CalculateResolution', 'Integration Options')

    def PyExec(self):

        self.setUp()
        outputs = []
        panel_outputs = self.getPropertyValue('PanelOutputWorkspaces')
        panel_output_groups = []
        for i in range(self.dimensionality):
            if self.sample[i] != EMPTY_TOKEN:
                self.reduce(i)
                outputs.append(self.output+ '_' + str(i + 1))
                panel_ws_group = panel_outputs + '_' + str(i + 1)
                if mtd.doesExist(panel_ws_group) and panel_outputs:
                    panel_output_groups.append(panel_ws_group)
            else:
                self.log().information('Skipping empty token run.')

        GroupWorkspaces(InputWorkspaces=outputs, OutputWorkspace=self.output)
        self.setProperty('OutputWorkspace', mtd[self.output])
        if self.output_sens:
            self.setProperty('SensitivityOutputWorkspace', mtd[self.output_sens])

        if panel_outputs and len(panel_output_groups) != 0:
            GroupWorkspaces(InputWorkspaces=panel_output_groups, OutputWorkspace=panel_outputs)
            self.setProperty('PanelOutputWorkspaces', mtd[panel_outputs])

    def reduce(self, i):

        [process_transmission_absorber, transmission_absorber_name] = needs_processing(self.atransmission, 'Absorber')
        self.progress.report('Processing transmission absorber')
        if process_transmission_absorber:
            SANSILLReduction(Run=self.atransmission,
                             ProcessAs='Absorber',
                             NormaliseBy=self.normalise,
                             OutputWorkspace=transmission_absorber_name)

        [process_transmission_beam, transmission_beam_name] = needs_processing(self.btransmission, 'Beam')
        self.progress.report('Processing transmission beam')
        flux_name = transmission_beam_name + '_Flux'
        if process_transmission_beam:
            SANSILLReduction(Run=self.btransmission,
                             ProcessAs='Beam',
                             NormaliseBy=self.normalise,
                             OutputWorkspace=transmission_beam_name,
                             BeamRadius=self.radius,
                             FluxOutputWorkspace=flux_name,
                             AbsorberInputWorkspace=transmission_absorber_name)

        [process_container_transmission, container_transmission_name] = needs_processing(self.ctransmission, 'Transmission')
        self.progress.report('Processing container transmission')
        if process_container_transmission:
            SANSILLReduction(Run=self.ctransmission,
                             ProcessAs='Transmission',
                             OutputWorkspace=container_transmission_name,
                             AbsorberInputWorkspace=transmission_absorber_name,
                             BeamInputWorkspace=transmission_beam_name,
                             NormaliseBy=self.normalise,
                             BeamRadius=self.radius)

        [process_sample_transmission, sample_transmission_name] = needs_processing(self.stransmission, 'Transmission')
        self.progress.report('Processing sample transmission')
        if process_sample_transmission:
            SANSILLReduction(Run=self.stransmission,
                             ProcessAs='Transmission',
                             OutputWorkspace=sample_transmission_name,
                             AbsorberInputWorkspace=transmission_absorber_name,
                             BeamInputWorkspace=transmission_beam_name,
                             NormaliseBy=self.normalise,
                             BeamRadius=self.radius)

        absorber = self.absorber[i] if len(self.absorber) == self.dimensionality else self.absorber[0]
        [process_absorber, absorber_name] = needs_processing(absorber, 'Absorber')
        self.progress.report('Processing absorber')
        if process_absorber:
            SANSILLReduction(Run=absorber,
                             ProcessAs='Absorber',
                             NormaliseBy=self.normalise,
                             OutputWorkspace=absorber_name)

        beam = self.beam[i] if len(self.beam) == self.dimensionality else self.beam[0]
        [process_beam, beam_name] = needs_processing(beam, 'Beam')
        flux_name = beam_name + '_Flux' if not self.flux[0] else ''
        self.progress.report('Processing beam')
        if process_beam:
            SANSILLReduction(Run=beam,
                             ProcessAs='Beam',
                             OutputWorkspace=beam_name,
                             NormaliseBy=self.normalise,
                             BeamRadius=self.radius,
                             AbsorberInputWorkspace=absorber_name,
                             FluxOutputWorkspace=flux_name)

        if self.flux[0]:
            flux = self.flux[i] if len(self.flux) == self.dimensionality else self.flux[0]
            [process_flux, flux_name] = needs_processing(flux, 'Flux')
            self.progress.report('Processing flux')
            if process_flux:
                SANSILLReduction(Run=flux,
                                 ProcessAs='Beam',
                                 OutputWorkspace=flux_name.replace('Flux', 'Beam'),
                                 NormaliseBy=self.normalise,
                                 BeamRadius=self.radius,
                                 AbsorberInputWorkspace=absorber_name,
                                 FluxOutputWorkspace=flux_name)

        container = self.container[i] if len(self.container) == self.dimensionality else self.container[0]
        [process_container, container_name] = needs_processing(container, 'Container')
        self.progress.report('Processing container')
        if process_container:
            SANSILLReduction(Run=container,
                             ProcessAs='Container',
                             OutputWorkspace=container_name,
                             AbsorberInputWorkspace=absorber_name,
                             BeamInputWorkspace=beam_name,
                             CacheSolidAngle=True,
                             TransmissionInputWorkspace=container_transmission_name,
                             ThetaDependent=self.theta_dependent,
                             NormaliseBy=self.normalise)

        # this is the default mask, the same for all the distance configurations
        [load_default_mask, default_mask_name] = needs_loading(self.default_mask, 'DefaultMask')
        self.progress.report('Loading default mask')
        if load_default_mask:
            LoadNexusProcessed(Filename=self.default_mask, OutputWorkspace=default_mask_name)

        # this is the beam stop mask, potentially different at each distance configuration
        mask = self.mask[i] if len(self.mask) == self.dimensionality else self.mask[0]
        [load_mask, mask_name] = needs_loading(mask, 'Mask')
        self.progress.report('Loading mask')
        if load_mask:
            LoadNexusProcessed(Filename=mask, OutputWorkspace=mask_name)

        sens_input = ''
        ref_input = ''
        if self.sensitivity:
            sens = self.sensitivity[i] if len(self.sensitivity) == self.dimensionality else self.sensitivity[0]
            [load_sensitivity, sensitivity_name] = needs_loading(sens, 'Sensitivity')
            sens_input = sensitivity_name
            self.progress.report('Loading sensitivity')
            if load_sensitivity:
                LoadNexusProcessed(Filename=sens, OutputWorkspace=sensitivity_name)
        if self.reference:
            reference = self.reference[i] if len(self.reference) == self.dimensionality else self.reference[0]
            [load_reference, reference_name] = needs_loading(reference, 'Reference')
            ref_input = reference_name
            self.progress.report('Loading reference')
            if load_reference:
                LoadNexusProcessed(Filename=reference, OutputWorkspace=reference_name)

        output = self.output + '_' + str(i + 1)
        [_, sample_name] = needs_processing(self.sample[i], 'Sample')
        self.progress.report('Processing sample at detector configuration '+str(i+1))

        SANSILLReduction(Run=self.sample[i],
                         ProcessAs='Sample',
                         OutputWorkspace=sample_name,
                         ReferenceInputWorkspace=ref_input,
                         AbsorberInputWorkspace=absorber_name,
                         BeamInputWorkspace=beam_name,
                         CacheSolidAngle=True,
                         ContainerInputWorkspace=container_name,
                         TransmissionInputWorkspace=sample_transmission_name,
                         MaskedInputWorkspace=mask_name,
                         DefaultMaskedInputWorkspace=default_mask_name,
                         SensitivityInputWorkspace=sens_input,
                         SensitivityOutputWorkspace=self.output_sens,
                         FluxInputWorkspace=flux_name,
                         NormaliseBy=self.normalise,
                         ThetaDependent=self.theta_dependent,
                         SampleThickness=self.getProperty('SampleThickness').value,
                         WaterCrossSection=self.getProperty('WaterCrossSection').value)
        panel_outputs = self.getPropertyValue('PanelOutputWorkspaces')
        panel_ws_group = panel_outputs + '_' + str(i + 1) if panel_outputs else ''
        SANSILLIntegration(InputWorkspace=sample_name,
                           OutputWorkspace=output,
                           OutputType=self.getPropertyValue('OutputType'),
                           CalculateResolution=self.getPropertyValue('CalculateResolution'),
                           DefaultQBinning=self.getPropertyValue('DefaultQBinning'),
                           BinningFactor=self.getProperty('BinningFactor').value,
                           OutputBinning=self.getPropertyValue('OutputBinning'),
                           NPixelDivision=self.getProperty('NPixelDivision').value,
                           NumberOfWedges=self.getProperty('NumberOfWedges').value,
                           WedgeAngle=self.getProperty('WedgeAngle').value,
                           WedgeOffset=self.getProperty('WedgeOffset').value,
                           AsymmetricWedges=self.getProperty('AsymmetricWedges').value,
                           PanelOutputWorkspaces=panel_ws_group)


AlgorithmFactory.subscribe(SANSILLAutoProcess)
