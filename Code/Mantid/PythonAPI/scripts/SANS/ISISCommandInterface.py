"""
    Command interface for ISIS SANS instruments.
    
    As commands in the common CommandInterface module are found to be ISIS-specific, 
    they should be moved here.
"""
import os

import SANSInsts
from CommandInterface import *
import ISISReductionSteps
import ISISReducer

#remove the following
DEL__FINDING_CENTRE_ = False


# disable plotting if running outside Mantidplot
try:
    from mantidplot import plotSpectrum, mergePlots
except ImportError:
    def plotSpectrum(a,b):
        pass
    def mergePlots(a,b):
        pass

try:
    from PyQt4.QtGui import qApp
    def appwidgets():
        return qApp.allWidgets()
except ImportError:
    def appwidgets():
        return []

_NOPRINT_ = False
_VERBOSE_ = False



def SetNoPrintMode(quiet = True):
    _NOPRINT_ = quiet

def SetVerboseMode(state):
    _VERBOSE_ = state

# Print a message and log it if the 
def _printMessage(msg, log = True):
    if log == True and _VERBOSE_ == True:
        mantid.sendLogMessage('::SANS::' + msg)
    if _NOPRINT_ == True: 
        return
    print msg
    
def issueWarning(msg):
    """
        Issues a Mantid message
        @param msg: message to be issued
    """
    ISISReductionSteps._issueWarning(msg)
                
def UserPath(path):
    ReductionSingleton().user_file_path = path
        
def SANS2D():
    instrument = SANSInsts.SANS2D()
        
    ReductionSingleton().set_instrument(instrument)

def LOQ():
    instrument = SANSInsts.LOQ()

    ReductionSingleton().set_instrument(instrument)
    
def Detector(det_name):
    ReductionSingleton().instrument.setDetector(det_name)
    
def Mask(details):
    _printMessage('Mask("' + details + '")')
    ReductionSingleton().mask(details)
    
def MaskFile(file_name):
    ReductionSingleton().user_settings = ISISReductionSteps.UserFile(
        file_name)
    ReductionSingleton().user_settings.execute(
        ReductionSingleton(), None)
    
def SetMonitorSpectrum(specNum, interp=False):
    ReductionSingleton().set_monitor_spectrum(specNum, interp)

def SuggestMonitorSpectrum(specNum, interp=False):  
    ReductionSingleton().suggest_monitor_spectrum(specNum, interp)
    
def SetTransSpectrum(specNum, interp=False):
    ReductionSingleton().set_trans_spectrum(specNum, interp)
      
def SetCentre(XVAL, YVAL):
    _printMessage('SetCentre(' + str(XVAL) + ',' + str(YVAL) + ')')
    ReductionSingleton().set_beam_finder(BaseBeamFinder(XVAL, YVAL))

def SetPhiLimit(phimin,phimax, phimirror=True):
    ReductionSingleton().set_phi_limit(phimin, phimax, phimirror)
    
def SetSampleOffset(value):
    ReductionSingleton().instrument.set_sample_offset(value)
    
def Gravity(flag):
    ReductionSingleton().to_Q.set_gravity(flag)
    
def TransFit(mode,lambdamin=None,lambdamax=None):
    ReductionSingleton().set_trans_fit(lambdamin, lambdamax, mode)
    
def AssignCan(can_run, reload = True, period = -1):
    ReductionSingleton().set_background(can_run, reload = reload, period = period)
    return ReductionSingleton().background_subtracter.assign_can(
                                                    ReductionSingleton())
    
def TransmissionSample(sample, direct, reload = True, period = -1):
    ReductionSingleton().set_trans_sample(sample, direct, reload = True, period = -1)
    return ReductionSingleton().samp_trans_load.execute(
                                            ReductionSingleton(), None)

def TransmissionCan(can, direct, reload = True, period = -1):
    _printMessage('TransmissionCan("' + can + '","' + direct + '")')
    ReductionSingleton().set_trans_can(can, direct, reload = True, period = -1)
    return ReductionSingleton().can_trans_load.execute(
                                            ReductionSingleton(), None)
    
def AssignSample(sample_run, reload = True, period = -1):
    _printMessage('AssignSample("' + sample_run + '")')

    ReductionSingleton().append_data_file(sample_run)
    ReductionSingleton().data_loader = ISISReductionSteps.LoadSample(
                                                            sample_run)
    ReductionSingleton().load_set_options(reload, period)

    sample_wksp, logs = ReductionSingleton().data_loader.execute(
                                            ReductionSingleton(), None)
    return sample_wksp, logs

def SetCentre(XVAL, YVAL):
    _printMessage('SetCentre(' + str(XVAL) + ',' + str(YVAL) + ')')
    SetBeamCenter(XVAL/1000.0, YVAL/1000.0)

def GetMismatchedDetList():
    """
        Return the list of mismatched detector names
    """
    return ReductionSingleton().instrument.get_marked_dets()

def WavRangeReduction(wav_start = None, wav_end = None, full_trans_wav = None):
    if not full_trans_wav is None:
        ReductionSingleton().full_trans_wav = full_trans_wav

    ReductionSingleton().to_wavelen.set_range(wav_start, wav_end)

    _printMessage('Running reduction for ' + str(ReductionSingleton().to_wavelen))
    

    if DEL__FINDING_CENTRE_ == True:
        final_workspace = wsname_cache.split('_')[0] + '_quadrants'
        if ReductionSingleton().to_Q.output_type == '1D':
                GroupIntoQuadrants(tmpWS, final_result, maskpt_rmin[0], maskpt_rmin[1], Q_REBIN)
                return
    #run the chain that has been created so far
    return ReductionSingleton.run()

    
def _SetWavelengthRange(start, end):
    ReductionSingleton().to_wavelen.set_range(start, end)


def Set1D():
    ReductionSingleton().to_Q.output_type = '1D'

def Set2D():
    ReductionSingleton().to_Q.output_type = '2D'

def SetSampleOffset(value):
    ReductionSingleton().instrument.set_sample_offset(value)

def SetRearEfficiencyFile(filename):
    rear_det = ReductionSingleton().instrument.getDetector('rear')
    rear_det.correction_file = filename

def SetFrontEfficiencyFile(filename):
    front_det = ReductionSingleton().instrument.getDetector('front')
    front_det.correction_file = filename

def SetDetectorFloodFile(filename):
    ReductionSingleton().flood_file.set_filename(filename)

def displayUserFile():
    print '-- Mask file defaults --'
    print ReductionSingleton().to_wavlen
    print ReductionSingleton().to_Q
    print correction_files()
    print '    direct beam file rear:',
    print ReductionSingleton().instrument.detector_file('rear')
    print '    direct beam file front:',
    print ReductionSingleton().instrument.detector_file('front')
    print ReductionSingleton().mask

def displayMaskFile():
    displayUserFile()

def displayGeometry():
    [x, y] = ReductionSingleton()._beam_finder.get_beam_center()
    print 'Beam centre: [' + str(x) + ',' + str(y) + ']'
    print ReductionSingleton().geometry

def SetPhiLimit(phimin,phimax, phimirror=True):
    maskStep = ReductionSingleton().get_mask()
    #a beam centre of [0,0,0] makes sense if the detector has been moved such that beam centre is at [0,0,0]
    maskStep.set_phi_limit(phimin, phimax, phimirror)
    
def LimitsPhi(phimin, phimax, use_mirror=True):
    '''
        !!DEPRECIATED by the function above, remove!!
        need to remove from SANSRunWindow.cpp
    '''
    if use_mirror :
        _printMessage("LimitsPHI(" + str(phimin) + ' ' + str(phimax) + 'use_mirror=True)')
        ReductionSingleton()._readLimitValues('L/PHI ' + str(phimin) + ' ' + str(phimax))
    else :
        _printMessage("LimitsPHI(" + str(phimin) + ' ' + str(phimax) + 'use_mirror=False)')
        ReductionSingleton()._readLimitValues('L/PHI/NOMIRROR ' + str(phimin) + ' ' + str(phimax))

def LimitsR(rmin, rmax):
    _printMessage('LimitsR(' + str(rmin) + ',' +str(rmax) + ')')
    settings = ReductionSingleton().user_settings
    settings._readLimitValues('L/R ' + str(rmin) + ' ' + str(rmax) + ' 1')

def LimitsWav(lmin, lmax, step, type):
    _printMessage('LimitsWav(' + str(lmin) + ',' + str(lmax) + ',' + str(step) + ','  + type + ')')
    _readLimitValues('L/WAV ' + str(lmin) + ' ' + str(lmax) + ' ' + str(step) + '/'  + type)

def LimitsQ(*args):
    # If given one argument it must be a rebin string
    if len(args) == 1:
        val = args[0]
        if type(val) == str:
            _printMessage("LimitsQ(" + val + ")")
            _readLimitValues("L/Q " + val)
        else:
            _issueWarning("LimitsQ can only be called with a single string or 4 values")
    elif len(args) == 4:
        qmin,qmax,step,step_type = args
        _printMessage('LimitsQ(' + str(qmin) + ',' + str(qmax) +',' + str(step) + ',' + str(step_type) + ')')
        _readLimitValues('L/Q ' + str(qmin) + ' ' + str(qmax) + ' ' + str(step) + '/'  + step_type)
    else:
        _issueWarning("LimitsQ called with " + str(len(args)) + " arguments, 1 or 4 expected.")

def LimitsQXY(qmin, qmax, step, type):
    _printMessage('LimitsQXY(' + str(qmin) + ',' + str(qmax) +',' + str(step) + ',' + str(type) + ')')
    _readLimitValues('L/QXY ' + str(qmin) + ' ' + str(qmax) + ' ' + str(step) + '/'  + type)

# These variables keep track of the centre coordinates that have been used so that we can calculate a relative shift of the
# detector
XVAR_PREV = 0.0
YVAR_PREV = 0.0
ITER_NUM = 0
RESIDUE_GRAPH = None

# Create a workspace with a quadrant value in it 
def _create_quadrant(reduced_ws, rawcount_ws, quadrant, xcentre, ycentre, q_bins, output):
    # Need to create a copy because we're going to mask 3/4 out and that's a one-way trip
    CloneWorkspace(reduced_ws,output)
    objxml = SANSUtility.QuadrantXML([xcentre, ycentre, 0.0], RMIN, RMAX, quadrant)
    # Mask out everything outside the quadrant of interest
    MaskDetectorsInShape(output,objxml)
    # Q1D ignores masked spectra/detectors. This is on the InputWorkspace, so we don't need masking of the InputForErrors workspace
    Q1D(output,rawcount_ws,output,q_bins,AccountForGravity=ReductionSingleton().to_Q.get_gravity())

    flag_value = -10.0
    ReplaceSpecialValues(InputWorkspace=output,OutputWorkspace=output,NaNValue=flag_value,InfinityValue=flag_value)
    if ReductionSingleton().to_Q.output_type == '1D':
        SANSUtility.StripEndZeroes(output, flag_value)

# Create 4 quadrants for the centre finding algorithm and return their names
def _group_into_quadrants(reduced_ws, final_result, xcentre, ycentre, q_bins):
    tmp = 'quad_temp_holder'
    pieces = ['Left', 'Right', 'Up', 'Down']
    to_group = ''
    counter = 0
    for q in pieces:
        counter += 1
        to_group += final_result + '_' + str(counter) + ','
        _create_quadrant(reduced_ws, final_result, q, xcentre, ycentre, q_bins, final_result + '_' + str(counter))

    # We don't need these now
    mantid.deleteWorkspace(reduced_ws)
# Calcluate the sum squared difference of the given workspaces. This assumes that a workspace with
# one spectrum for each of the quadrants. The order should be L,R,U,D.
def CalculateResidue():
    global XVAR_PREV, YVAR_PREV, RESIDUE_GRAPH
    yvalsA = mtd.getMatrixWorkspace('Left').readY(0)
    yvalsB = mtd.getMatrixWorkspace('Right').readY(0)
    qvalsA = mtd.getMatrixWorkspace('Left').readX(0)
    qvalsB = mtd.getMatrixWorkspace('Right').readX(0)
    qrange = [len(yvalsA), len(yvalsB)]
    nvals = min(qrange)
    residueX = 0
    indexB = 0
    for indexA in range(0, nvals):
        if qvalsA[indexA] < qvalsB[indexB]:
            mantid.sendLogMessage("::SANS::LR1 "+str(indexA)+" "+str(indexB))
            continue
        elif qvalsA[indexA] > qvalsB[indexB]:
            while qvalsA[indexA] > qvalsB[indexB]:
                mantid.sendLogMessage("::SANS::LR2 "+str(indexA)+" "+str(indexB))
                indexB += 1
        if indexA > nvals - 1 or indexB > nvals - 1:
            break
        residueX += pow(yvalsA[indexA] - yvalsB[indexB], 2)
        indexB += 1

    yvalsA = mtd.getMatrixWorkspace('Up').readY(0)
    yvalsB = mtd.getMatrixWorkspace('Down').readY(0)
    qvalsA = mtd.getMatrixWorkspace('Up').readX(0)
    qvalsB = mtd.getMatrixWorkspace('Down').readX(0)
    qrange = [len(yvalsA), len(yvalsB)]
    nvals = min(qrange)
    residueY = 0
    indexB = 0
    for indexA in range(0, nvals):
        if qvalsA[indexA] < qvalsB[indexB]:
            mantid.sendLogMessage("::SANS::UD1 "+str(indexA)+" "+str(indexB))
            continue
        elif qvalsA[indexA] > qvalsB[indexB]:
            while qvalsA[indexA] > qvalsB[indexB]:
                mantid.sendLogMessage("::SANS::UD2 "+str(indexA)+" "+str(indexB))
                indexB += 1
        if indexA > nvals - 1 or indexB > nvals - 1:
            break
        residueY += pow(yvalsA[indexA] - yvalsB[indexB], 2)
        indexB += 1
                        
    try :
        if RESIDUE_GRAPH is None or (not RESIDUE_GRAPH in appwidgets()):
            RESIDUE_GRAPH = plotSpectrum('Left', 0)
            mergePlots(RESIDUE_GRAPH, plotSpectrum(['Right','Up'],0))
            mergePlots(RESIDUE_GRAPH, plotSpectrum(['Down'],0))
        RESIDUE_GRAPH.activeLayer().setTitle("Itr " + str(ITER_NUM)+" "+str(XVAR_PREV*1000.)+","+str(YVAR_PREV*1000.)+" SX "+str(residueX)+" SY "+str(residueY))
    except :
        #if plotting is not available it probably means we are running outside a GUI, in which case do everything but don't plot
        pass
        
    mantid.sendLogMessage("::SANS::Itr: "+str(ITER_NUM)+" "+str(XVAR_PREV*1000.)+","+str(YVAR_PREV*1000.)+" SX "+str(residueX)+" SY "+str(residueY))              
    return residueX, residueY

def PlotResult(workspace):
    if ReductionSingleton().to_Q.output_type == '1D':
        plotSpectrum(workspace,0)
    else:
        qti.app.mantidUI.importMatrixWorkspace(workspace).plotGraph2D()

##################### View mask details #####################################################

def ViewCurrentMask():
    ReductionSingleton().ViewCurrentMask()

# Print a test script for Colette if asked
def createColetteScript(inputdata, format, reduced, centreit , plotresults, csvfile = '', savepath = ''):
    script = ''
    if csvfile != '':
        script += '[COLETTE]  @ ' + csvfile + '\n'
    file_1 = inputdata['sample_sans'] + format
    script += '[COLETTE]  ASSIGN/SAMPLE ' + file_1 + '\n'
    file_1 = inputdata['sample_trans'] + format
    file_2 = inputdata['sample_direct_beam'] + format
    if file_1 != format and file_2 != format:
        script += '[COLETTE]  TRANSMISSION/SAMPLE/MEASURED ' + file_1 + ' ' + file_2 + '\n'
    file_1 = inputdata['can_sans'] + format
    if file_1 != format:
        script +='[COLETTE]  ASSIGN/CAN ' + file_1 + '\n'
    file_1 = inputdata['can_trans'] + format
    file_2 = inputdata['can_direct_beam'] + format
    if file_1 != format and file_2 != format:
        script += '[COLETTE]  TRANSMISSION/CAN/MEASURED ' + file_1 + ' ' + file_2 + '\n'
    if centreit:
        script += '[COLETTE]  FIT/MIDDLE'
    # Parameters
    script += '[COLETTE]  LIMIT/RADIUS ' + str(ReductionSingleton().mask.min_radius)
    script += ' ' + str(ReductionSingleton().mask.max_radius) + '\n'
    script += '[COLETTE]  LIMIT/WAVELENGTH ' + ReductionSingleton().to_wavelen.get_range() + '\n'
    if DWAV <  0:
        script += '[COLETTE]  STEP/WAVELENGTH/LOGARITHMIC ' + str(ReductionSingleton().to_wavelen.w_step)[1:] + '\n'
    else:
        script += '[COLETTE]  STEP/WAVELENGTH/LINEAR ' + str(ReductionSingleton().to_wavelen.w_step) + '\n'
    # For the moment treat the rebin string as min/max/step
    qbins = q_REBEIN.split(",")
    nbins = len(qbins)
    if ReductionSingleton().to_Q.output_type == '1D':
        script += '[COLETTE]  LIMIT/Q ' + str(qbins[0]) + ' ' + str(qbins[nbins-1]) + '\n'
        dq = float(qbins[1])
        if dq <  0:
            script += '[COLETTE]  STEP/Q/LOGARITHMIC ' + str(dq)[1:] + '\n'
        else:
            script += '[COLETTE]  STEP/Q/LINEAR ' + str(dq) + '\n'
    else:
        script += '[COLETTE]  LIMIT/QXY ' + str(0.0) + ' ' + str(ReductionSingleton().QXY2) + '\n'
        if ReductionSingleton().DQXY <  0:
            script += '[COLETTE]  STEP/QXY/LOGARITHMIC ' + str(ReductionSingleton().DQXY)[1:] + '\n'
        else:
            script += '[COLETTE]  STEP/QXY/LINEAR ' + str(ReductionSingleton().DQXY) + '\n'
    
    # Correct
    script += '[COLETTE] CORRECT\n'
    if plotresults:
        script += '[COLETTE]  DISPLAY/HISTOGRAM ' + reduced + '\n'
    if savepath != '':
        script += '[COLETTE]  WRITE/LOQ ' + reduced + ' ' + savepath + '\n'
        
    return script

ReductionSingleton.clean(ISISReducer.ISISReducer)

#this is like a #define I'd like to get rid of it because it means nothing here
DefaultTrans = 'True'
NewTrans = 'False'