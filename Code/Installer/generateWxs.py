# Automatic creation of installer source file (.wxs)
import os
import os.path
import xml
import xml.dom.minidom
import msilib

QTDIR = 'C:/qt/bin'
#QTDIR = 'C:/Qt/4_4_0'

globalFileCount = 0

# Adds directory longName to parent.
# parent is a python variable (not string) representing an xml element
# Id, name, and longName are strings
def addDirectory(Id,name,longName,parent):
    e = doc.createElement('Directory')
    e.setAttribute('Id',Id)
    e.setAttribute('Name',name)
    if name != longName:
        e.setAttribute('LongName',longName)
    parent.appendChild(e)
    return e

def addFile(Id,name,longName,source,vital,parent):
    e = doc.createElement('File')
    e.setAttribute('Id',Id)
    e.setAttribute('Name',name)
    e.setAttribute('LongName',longName)
    e.setAttribute('DiskId','1')
    e.setAttribute('Source',source)
    e.setAttribute('Vital',vital)
    parent.appendChild(e)
    return e

def addFileV(Id,name,longName,source,parent):
    return addFile(Id,name,longName,source,'yes',parent)

def addFileN(Id,name,longName,source,parent):
    return addFile(Id,name,longName,source,'no',parent)

def addComponent(Id,guid,parent):
    e = doc.createElement('Component')
    e.setAttribute('Id',Id)
    e.setAttribute('Guid',guid)
    parent.appendChild(e)
    return e

# adds all dlls from location to parent.
# rules are applied to exclude debug libraries
# name is a short name to which a number will be added
def addDlls(location,name,parent):
    print 'Include dlls from',os.path.abspath(location);
    sdlls = os.listdir(location);
    i = 0
    for fil in sdlls:
        fn = fil.split('.')
        if len(fn) == 2 and fn[1] == 'dll':
            fn0 = fn[0].replace('-','_')
            if not ((fil.find('-gd-') >= 0) or
                    (fil.find('d.dll')>=0 and fil.replace('d.dll','.dll') in sdlls) or
                    (fil.find('d4.dll')>=0 and fil.replace('d4.dll','4.dll') in sdlls)):
                print fn[0]+'.'+fn[1]
                addFileV(fn0+fn[1],name+str(i),fil,location+'/'+fil,parent)
            i += 1

def addAllFiles(location,name,parent):
    print 'Include files from',os.path.abspath(location);
    sfiles = os.listdir(location);
    i = 0
    for fil in sfiles:
        print fil
        fn = fil.replace('-','_')
        fn = fn.replace('+','_')
        if (fil.find('.svn') < 0 and os.path.isfile(location+'/'+fil)):
            addFileV(name+'_'+fn+'_file',name+str(i),fil,location+'/'+fil,parent)
            i += 1

def addAllFilesExt(location,name,ext,parent):
    print 'Include files from',os.path.abspath(location);
    sfiles = os.listdir(location);
    i = 0
    for fil in sfiles:
        fn = fil.replace('-','_')
        fn = fn.replace('+','_')
        if (fil.find('.svn') < 0 and fil.endswith('.'+ext) > 0):
            print fil
            addFileV(name+'_'+fn+'_file',name+str(i),fil,location+'/'+fil,parent)
            i += 1

def addFeature(Id,title,description,level,parent):
    e = doc.createElement('Feature')
    e.setAttribute('Id',Id)
    e.setAttribute('Title',title)
    e.setAttribute('Description',description)
    e.setAttribute('Level',level)
    parent.appendChild(e)
    return e

def addRootFeature(Id,title,description,level,parent):
    e = doc.createElement('Feature')
    e.setAttribute('Id',Id)
    e.setAttribute('Title',title)
    e.setAttribute('Description',description)
    e.setAttribute('Level',level)
    e.setAttribute('Display','expand')
    e.setAttribute('ConfigurableDirectory','INSTALLDIR')
    parent.appendChild(e)
    return e

def addCRef(Id,parent):
    e = doc.createElement('ComponentRef')
    e.setAttribute('Id',Id)
    parent.appendChild(e)

# adds to parent an element tag with dictionary of attributes attr
def addTo(parent,tag,attr):
    e = doc.createElement(tag)
    for name,value in attr.iteritems():
        e.setAttribute(name,value)
    parent.appendChild(e)
    return e

def fileSearch(Id,name,parent):
    p = addTo(parent,'Property',{'Id':Id})
    e = addTo(p,'FileSearch',{'Id':Id+'_search','LongName':name})
    return e
    
def addText(text,parent):
    e = doc.createTextNode(text)
    parent.appendChild(e)
    return e

# Copies files in nested folders from location to parent directory
# Returns a list of component names to be used in addCRefs
def addCompList(Id,location,name,parent):
    global globalFileCount
    directory = addDirectory(Id+'_dir','dir',name,parent)
    lst = []
    idir = 0
#    ifil = 0
    comp = addComponent(Id,msilib.gen_uuid(),directory)
    lst.append(Id)
    files = os.listdir(location)
    for fil in files:
        if ( fil.find('.svn') < 0 and os.path.isdir(location+'/'+fil) ):
            idir += 1
            lst = lst + addCompList(Id+'_'+str(idir), location+'/'+fil, fil, directory)
        elif fil.find('.svn') < 0:
            globalFileCount += 1
            ifil = globalFileCount
            fn = fil.replace('-','_')
            fn = fn.replace('+','_')
            fileId = 'd'+fn+'_file'+str(ifil)
            fileName = 'file'+str(ifil)
            fileLongName = fil
            addFileV(fileId,fileName,fileLongName,location+'/'+fil,comp)
    return lst

def addCRefs(lstId,parent):
    for Id in lstId:
        e = doc.createElement('ComponentRef')
        e.setAttribute('Id',Id)
        parent.appendChild(e)

doc = xml.dom.minidom.Document()
#doc.encoding('Windows-1252')
wix = doc.createElement('Wix')
wix.setAttribute('xmlns','http://schemas.microsoft.com/wix/2003/01/wi')
doc.appendChild(wix)

Product = doc.createElement('Product')
Product.setAttribute('Name','Mantid')
Product.setAttribute('Id','{CA88C1C7-AEB8-4bd6-A62C-9C436FA31211}')
Product.setAttribute('Language','1033')
Product.setAttribute('Codepage','1252')
Product.setAttribute('Version','1.0.0')
Product.setAttribute('Manufacturer','STFC Rutherford Appleton Laboratories')
wix.appendChild(Product)

Package = doc.createElement('Package')
Package.setAttribute('Id','????????-????-????-????-????????????')
Package.setAttribute('Keywords','Installer')
Package.setAttribute('Description','Mantid Installer')
#Package.setAttribute('Comments','')
Package.setAttribute('Manufacturer','STFC Rutherford Appleton Laboratories')
Package.setAttribute('InstallerVersion','100')
Package.setAttribute('Languages','1033')
Package.setAttribute('Compressed','yes')
Package.setAttribute('SummaryCodepage','1252')
Product.appendChild(Package)

Media = doc.createElement('Media')
Media.setAttribute('Id','1')
Media.setAttribute('Cabinet','Mantid.cab')
Media.setAttribute('EmbedCab','yes')
Media.setAttribute('DiskPrompt','CD-ROM #1')
Product.appendChild(Media)

Prop = doc.createElement('Property')
Prop.setAttribute('Id','DiskPrompt')
Prop.setAttribute('Value','Mantid Installation')
Product.appendChild(Prop)

Prop = doc.createElement('Property')
Prop.setAttribute('Id','PYTHON_25_DIR_EXISTS')
Product.appendChild(Prop)
DS = doc.createElement('DirectorySearch')
DS.setAttribute('Id','CheckPyDir')
DS.setAttribute('Path','C:\\Python25')
DS.setAttribute('Depth','0')
Prop.appendChild(DS)

Cond = doc.createElement('Condition')
Cond.setAttribute('Message','Mantid requires Python 2.5 to be installed on your machine. It can be downloaded and installed from http://www.python.org/download/')
Cond.appendChild(doc.createTextNode('PYTHON_25_DIR_EXISTS'))
Product.appendChild(Cond)

TargetDir = addDirectory('TARGETDIR','SourceDir','SourceDir',Product)
InstallDir = addDirectory('INSTALLDIR','MInstall','MantidInstall',TargetDir)
binDir = addDirectory('MantidBin','bin','bin',InstallDir)

MantidDlls = addComponent('MantidDLLs','{FABC0481-C18D-415e-A0B1-CCB76C35FBE8}',binDir)
addFileV('MantidProperties','Mantid.pro','Mantid.properties','../Mantid/release/Mantid.properties',MantidDlls)
MantidScript = addFileV('MantidScript','MScr.bat','MantidScript.bat','../Mantid/PythonAPI/MantidScript.bat',MantidDlls)
addTo(MantidScript,'Shortcut',{'Id':'startmenuMantidScript','Directory':'ProgramMenuDir','Name':'Script','LongName':'Mantid Script','WorkingDirectory':'binDir'})
addFileV('MantidStartup','MStart.py','MantidStartup.py','../Mantid/PythonAPI/MantidStartup.py',MantidDlls)
addFileV('MantidPythonAPI','MPAPI.pyd','MantidPythonAPI.pyd','../Mantid/Bin/Shared/MantidPythonAPI.dll',MantidDlls)
addDlls('../Mantid/Bin/Shared','SDll',MantidDlls)
addDlls('../Mantid/Bin/Plugins','PnDll',MantidDlls)
addDlls('../Third_Party/lib/win32','3dDll',MantidDlls)
addAllFiles('toget/MSVCruntime','ms',MantidDlls)

QTIPlot = addComponent('QTIPlot','{03ABDE5C-9084-4ebd-9CF8-31648BEFDEB7}',binDir)
addDlls(QTDIR+'/bin','qt',QTIPlot)
QTIPlotEXE = addFileV('QTIPlotEXE','qtiplot.exe','qtiplot.exe','../qtiplot/qtiplot/qtiplot.exe',QTIPlot)
startmenuQTIPlot = addTo(QTIPlotEXE,'Shortcut',{'Id':'startmenuQTIPlot','Directory':'ProgramMenuDir','Name':'QTIPlot','WorkingDirectory':'binDir'})
desktopQTIPlot = addTo(QTIPlotEXE,'Shortcut',{'Id':'desktopQTIPlot','Directory':'DesktopFolder','Name':'QTIPlot','WorkingDirectory':'binir'})
addAllFiles('toget/pyc','pyc',QTIPlot)
if (QTDIR == 'C:/Qt/4_4_0'):
    manifestFile = addFileV('qtiplot_manifest','qtiexe.man','qtiplot.exe.manifest','../qtiplot/qtiplot/qtiplot.exe.manifest',QTIPlot)

addTo(MantidDlls,'RemoveFile',{'Id':'LogFile','On':'uninstall','Name':'mantid.log'})

pluginsDir = addDirectory('PluginsDir','plugins','plugins',InstallDir)
Plugins = addComponent('Plugins','{EEF0B4C9-DE52-4f99-A8D0-9D3C3941FA73}',pluginsDir)
addTo(Plugins,'CreateFolder',{})

documentsDir = addDirectory('DocumentsDir','docs','docs',InstallDir)
Documents = addComponent('Documents','{C16B2B59-17C8-4cc9-8A7F-16254EB8B2F4}',documentsDir)
addTo(Documents,'CreateFolder',{})

logsDir = addDirectory('LogsDir','logs','logs',InstallDir)
Logs = addComponent('Logs','{0918C9A4-3481-4f21-B941-983BE21F9674}',logsDir)
addTo(Logs,'CreateFolder',{})

#-------------------  Includes  -------------------------------------
includeDir = addDirectory('IncludeDir','include','include',InstallDir)
includeMantidAlgorithmsDir = addDirectory('IncludeMantidAlgorithmsDir','MAlgs','MantidAlgorithms',includeDir)
IncludeMantidAlgorithms = addComponent('IncludeMantidAlgorithms','{EDB85D81-1CED-459a-BF87-E148CEE6F9F6}',includeMantidAlgorithmsDir)
addAllFiles('../Mantid/includes/MantidAlgorithms','alg',IncludeMantidAlgorithms)

includeMantidAPIDir = addDirectory('IncludeMantidAPIDir','MAPI','MantidAPI',includeDir)
IncludeMantidAPI = addComponent('IncludeMantidAPI','{4761DDF6-813C-4470-8852-98CB9A69EBC9}',includeMantidAPIDir)
addAllFiles('../Mantid/includes/MantidAPI','api',IncludeMantidAPI)

includeMantidDataHandlingDir = addDirectory('IncludeMantidDataHandlingDir','MDH','MantidDataHandling',includeDir)
IncludeMantidDataHandling = addComponent('IncludeMantidDataHandling','{DDD2DD4A-9A6A-4181-AF66-891B99DF8FFE}',includeMantidDataHandlingDir)
addAllFiles('../Mantid/includes/MantidDataHandling','dh',IncludeMantidDataHandling)

includeMantidDataObjectsDir = addDirectory('IncludeMantidDataObjectsDir','MDO','MantidDataObjects',includeDir)
IncludeMantidDataObjects = addComponent('IncludeMantidDataObjects','{06445843-7E74-4457-B02E-4850B4911438}',includeMantidDataObjectsDir)
addAllFiles('../Mantid/includes/MantidDataObjects','do',IncludeMantidDataObjects)

includeMantidGeometryDir = addDirectory('IncludeMantidGeometryDir','GEO','MantidGeometry',includeDir)
IncludeMantidGeometry = addComponent('IncludeMantidGeometry','{AF39B1A0-5068-4f2d-B9B9-D69926404686}',includeMantidGeometryDir)
addAllFiles('../Mantid/includes/MantidGeometry','geo',IncludeMantidGeometry)

includeMantidKernelDir = addDirectory('IncludeMantidKernelDir','KER','MantidKernel',includeDir)
IncludeMantidKernel = addComponent('IncludeMantidKernel','{AF40472B-5822-4ff6-8E05-B4DA5224AA87}',includeMantidKernelDir)
addAllFiles('../Mantid/includes/MantidKernel','ker',IncludeMantidKernel)

includeMantidNexusDir = addDirectory('IncludeMantidNexusDir','NEX','MantidNexus',includeDir)
IncludeMantidNexus = addComponent('IncludeMantidNexus','{BAC18721-6DF1-4870-82FD-2FB37260AE35}',includeMantidNexusDir)
addAllFiles('../Mantid/includes/MantidNexus','nex',IncludeMantidNexus)

includeMantidPythonAPIDir = addDirectory('IncludeMantidPythonAPIDir','PAPI','MantidPythonAPI',includeDir)
IncludeMantidPythonAPI = addComponent('IncludeMantidPythonAPI','{052A15D4-97A0-4ce5-A872-E6871485E734}',includeMantidPythonAPIDir)
addAllFiles('../Mantid/includes/MantidPythonAPI','papi',IncludeMantidPythonAPI)

includeMantidServicesDir = addDirectory('IncludeMantidServicesDir','SERV','MantidServices',includeDir)
IncludeMantidServices = addComponent('IncludeMantidServices','{7EC9AF3A-3907-42bf-8542-DCBFFD9ECFE5}',includeMantidServicesDir)
addAllFiles('../Mantid/includes/MantidServices','serv',IncludeMantidServices)

boostList = addCompList('boost','../Third_Party/include/boost','boost',includeDir)
#-------------------  end of Includes ---------------------------------------

sconsList = addCompList('scons','../Third_Party/src/scons-local','scons-local',InstallDir)

tempDir = addDirectory('TempDir','temp','temp',InstallDir)
Temp = addComponent('Temp','{02D25B60-A114-4f2a-A211-DE88CF648C61}',tempDir)
addTo(Temp,'CreateFolder',{})

dataDir = addDirectory('DataDir','data','data',InstallDir)
Data = addComponent('Data','{6D9A0A53-42D5-46a5-8E88-6BB4FB7A5FE1}',dataDir)
addTo(Data,'CreateFolder',{})

#-------------------  Source  ------------------------------------------
sourceDir = addDirectory('SourceDir','source','source',InstallDir)

sourceMantidAlgorithmsDir = addDirectory('SourceMantidAlgorithmsDir','MAlgs','MantidAlgorithms',sourceDir)
SourceMantidAlgorithms = addComponent('SourceMantidAlgorithms','{C96FA514-351A-4e60-AC4F-EF07216BBDC9}',sourceMantidAlgorithmsDir)
addAllFilesExt('../Mantid/Algorithms/src','alg','cpp',SourceMantidAlgorithms)

sourceMantidAPIDir = addDirectory('SourceMantidAPIDir','MAPI','MantidAPI',sourceDir)
SourceMantidAPI = addComponent('SourceMantidAPI','{3186462A-E033-4682-B992-DA80BAF457F2}',sourceMantidAPIDir)
addAllFilesExt('../Mantid/API/src','api','cpp',SourceMantidAPI)

sourceMantidDataHandlingDir = addDirectory('SourceMantidDataHandlingDir','Mdh','MantidDataHandling',sourceDir)
SourceMantidDataHandling = addComponent('SourceMantidDataHandling','{3DE8C8E7-86F1-457f-8933-149AD79EA9D7}',sourceMantidDataHandlingDir)
addAllFilesExt('../Mantid/DataHandling/src','dh','cpp',SourceMantidDataHandling)

sourceMantidDataObjectsDir = addDirectory('SourceMantidDataObjectsDir','Mdo','MantidDataObjects',sourceDir)
SourceMantidDataObjects = addComponent('SourceMantidDataObjects','{0C071065-8E0C-4e9c-996E-454692803E7F}',sourceMantidDataObjectsDir)
addAllFilesExt('../Mantid/DataObjects/src','dh','cpp',SourceMantidDataObjects)

sourceMantidGeometryDir = addDirectory('SourceMantidGeometryDir','MGeo','MantidGeometry',sourceDir)
SourceMantidGeometry = addComponent('SourceMantidGeometry','{949C5B12-7D4B-4a8a-B132-718F6AEA9E69}',sourceMantidGeometryDir)
addAllFilesExt('../Mantid/Geometry/src','geo','cpp',SourceMantidGeometry)

sourceMantidKernelDir = addDirectory('SourceMantidKernelDir','MKer','MantidKernel',sourceDir)
SourceMantidKernel = addComponent('SourceMantidKernel','{B7126F68-544C-4e50-9438-E0D6F6155D82}',sourceMantidKernelDir)
addAllFilesExt('../Mantid/Kernel/src','ker','cpp',SourceMantidKernel)

sourceMantidNexusDir = addDirectory('SourceMantidNexusDir','MNex','MantidNexus',sourceDir)
SourceMantidNexus = addComponent('SourceMantidNexus','{35AABB59-CDE3-49bf-9F96-7A1AFB72FD2F}',sourceMantidNexusDir)
addAllFilesExt('../Mantid/Nexus/src','nex','cpp',SourceMantidNexus)

sourceMantidPythonAPIDir = addDirectory('SourceMantidPythonAPIDir','MPAPI','MantidPythonAPI',sourceDir)
SourceMantidPythonAPI = addComponent('SourceMantidPythonAPI','{CACED707-92D7-47b9-8ABC-378275D99082}',sourceMantidPythonAPIDir)
addAllFilesExt('../Mantid/PythonAPI/src','papi','cpp',SourceMantidPythonAPI)

#----------------- end of Source ---------------------------------------

#----------------- User Algorithms -------------------------------------
UserAlgorithmsDir = addDirectory('UserAlgorithmsDir','UAlgs','UserAlgorithms',InstallDir)
UserAlgorithms = addComponent('UserAlgorithms',msilib.gen_uuid(),UserAlgorithmsDir)
addAllFilesExt('../Mantid/UserAlgorithms','ualg','cpp',UserAlgorithms)
addAllFilesExt('../Mantid/UserAlgorithms','ualg','h',UserAlgorithms)
addFileV('Sconstruct','Sconstr','Sconstruct','toget/UserAlgorithms/Sconstruct',UserAlgorithms)
addFileV('build_bat','build.bat','build.bat','toget/UserAlgorithms/build.bat',UserAlgorithms)
addFileV('MantidKernel_lib','MKernel.lib','MantidKernel.lib','../Mantid/Kernel/lib/MantidKernel.lib',UserAlgorithms)
addFileV('MantidAPI_lib','MAPI.lib','MantidAPI.lib','../Mantid/API/lib/MantidAPI.lib',UserAlgorithms)
addFileV('MantidDataObjects_lib','MDObject.lib','MantidDataObjects.lib','../Mantid/DataObjects/lib/MantidDataObjects.lib',UserAlgorithms)
addFileV('MantidGeometry_lib','MGeo.lib','MantidGeometry.lib','../Mantid/Geometry/lib/MantidGeometry.lib',UserAlgorithms)
addFileV('boost_filsystem_lib','boost_fs.lib','libboost_filesystem-vc80-mt-1_34_1.lib','../Third_Party/lib/win32/libboost_filesystem-vc80-mt-1_34_1.lib',UserAlgorithms)

#--------------- Python ------------------------------------------------

Python25Dir = addDirectory('Python25Dir','Python25','Python25',TargetDir)
LibDir = addDirectory('LibDir','Lib','Lib',Python25Dir)
SitePackagesDir = addDirectory('SitePackagesDir','sitepack','site-packages',LibDir)
PyQtDir = addDirectory('PyQtDir','PyQt4','PyQt4',SitePackagesDir)
Sip = addComponent('Sip','{A051F48C-CA96-4cd5-B936-D446CBF67588}',SitePackagesDir)
addAllFiles('toget/sip','sip',Sip)
PyQt = addComponent('PyQt','{18028C0B-9DF4-48f6-B8FC-DE195FE994A0}',PyQtDir)
addAllFiles('toget/PyQt4','PyQt',PyQt)
#-------------------------- Scripts ------------------------------------
ScriptsDir = addDirectory('ScriptsDir','scripts','scripts',InstallDir)
Scripts = addComponent('Scripts',msilib.gen_uuid(),ScriptsDir)
addAllFiles('../Mantid/PythonAPI/scripts','scr',Scripts)
#-----------------------------------------------------------------------

ProgramMenuFolder = addDirectory('ProgramMenuFolder','PMenu','Programs',TargetDir)
ProgramMenuDir = addDirectory('ProgramMenuDir','Mantid','Mantid',ProgramMenuFolder)

DesktopFolder = addDirectory('DesktopFolder','Desktop','Desktop',TargetDir)

#-----------------------------------------------------------------------
Py25Exists = addTo(Product,'Property',{'Id':'DIREXISTS'})
addTo(Py25Exists,'DirectorySearch',{'Id':'CheckDir','Path':'C:\Python25\Lib\site-packages\PyQt4','Depth':'0'})

Complete = addRootFeature('Complete','Mantid','The complete package','1',Product)
MantidExec = addFeature('MantidExecAndDlls','Mantid binaries','The main executable.','1',Complete)
addCRef('MantidDLLs',MantidExec)
addCRef('Plugins',MantidExec)
addCRef('Documents',MantidExec)
addCRef('Logs',MantidExec)
addCRef('Scripts',MantidExec)
addCRef('IncludeMantidAlgorithms',MantidExec)
addCRef('IncludeMantidAPI',MantidExec)
addCRef('IncludeMantidDataHandling',MantidExec)
addCRef('IncludeMantidDataObjects',MantidExec)
addCRef('IncludeMantidGeometry',MantidExec)
addCRef('IncludeMantidKernel',MantidExec)
addCRef('IncludeMantidNexus',MantidExec)
addCRef('IncludeMantidPythonAPI',MantidExec)
addCRef('IncludeMantidServices',MantidExec)
addCRef('Temp',MantidExec)
addCRef('Data',MantidExec)

addCRefs(sconsList,MantidExec)
addCRefs(boostList,MantidExec)
addCRef('UserAlgorithms',MantidExec)

QTIPlotExec = addFeature('QTIPlotExec','QtiPlot','QtiPlot','1',MantidExec)
addCRef('QTIPlot',QTIPlotExec)

PyQtF = addFeature('PyQtF','PyQt4','PyQt4','1',MantidExec)
addCRef('Sip',PyQtF)
addCRef('PyQt',PyQtF)

SourceFiles = addFeature('SourceFiles','SourceFiles','SourceFiles','1000',Complete)
addCRef('SourceMantidAlgorithms',SourceFiles)
addCRef('SourceMantidAPI',SourceFiles)
addCRef('SourceMantidDataHandling',SourceFiles)
addCRef('SourceMantidDataObjects',SourceFiles)
addCRef('SourceMantidGeometry',SourceFiles)
addCRef('SourceMantidKernel',SourceFiles)
addCRef('SourceMantidNexus',SourceFiles)
addCRef('SourceMantidPythonAPI',SourceFiles)

addTo(Product,'UIRef',{'Id':'WixUI_Mondo'})
addTo(Product,'UIRef',{'Id':'WixUI_ErrorProgressText'})

f = open('tmp.wxs','w')
doc.writexml(f)
f.close()
