
# Required parameters
#
#	TargetDir		Project subfolder with files to be copied (e.g. WinDebug). Define it as $(OutDir).
#
#	TargetName		Base name of the files to be copied. Define it as $(TargetName).
#
# Optional parameters
#
#	ProjDir			Project folder (default is ".", i.e. current folder)
#
#	HierarchyLevel	Depth of ProjDir relative to ROOT. Default is 1. Used to determine LibRelativePath.
#
#	LibRelativePath	Path to lib destination relative to $(ProjDir). Default depends on HierarchyLevel.
#
#	DontCopyLib		If defined, lib files are not copied.
#
#	RelativeBRD		Path to executables destination relative to $(RESULTS_ROOT). Default is $(TargetDir)
#
#	CustomCopy		Name of the *.mak file with additional rules.
#					It wll be included by !include "$(CustomCopy)" directive.
#					File can modify "targets" macro (add additional space-delimeted 
#					targets at the end) and use $(BRD), $(LBD), $(SRD), $(ProjDir),
#					$(ROOT), $(TargetName) macros to define copy rules.
#
#	Platform		Platform name, e.g. x64. Used as suffix for library directory.
#
# Internal macros
#
#	SRD				Folder with files to be copied, $(ProjDir)\$(TargetDir)
#
#	LBD				Destination for LIBs, $(ProjDir)\$(LibRelativePath)
#
#	BRD				Destination for executables, PDBs, etc., $(RESULTS_ROOT)\$(RelativeBRD)
#
# Environment
#
#	ROOT			Build root, required
#	RESULTS_ROOT	Results root, default is $(ROOT)
#	COPYPDB			If set to 0, PDBs are not copied.
#

# main pseudorule
all: copyAllFiles
#all: debugPrint
	
!if !defined( TargetDir ) 
!	error Undefined TargetDir
!endif

!if !defined( TargetName ) 
!	error Undefined TargetName
!endif

# ROOT
!ifdef RESULTS_ROOT
ROOT=$(RESULTS_ROOT)
!endif

!if !defined( ROOT ) 
!	error Undefined ROOT
!endif

# project directory
!ifndef ProjDir
ProjDir=.
!endif

# build result directory
!ifndef RelativeBRD
RelativeBRD=$(TargetDir)
!endif

!ifndef BRD
BRD=$(ROOT)\$(RelativeBRD)
!endif

# copy source directory
!ifndef SRD
SRD=$(ProjDir)\$(TargetDir)
!endif

!ifndef HierarchyLevel
HierarchyLevel=1
!endif

# library output directory path relative to the project
!ifndef LibRelativePath
!	if "$(HierarchyLevel)" == "1"
LibRelativePath=lib
!	else if "$(HierarchyLevel)" == "2"
LibRelativePath=..\lib
!	else if "$(HierarchyLevel)" == "3"
LibRelativePath=..\..\lib
!	else if "$(HierarchyLevel)" == "4"
LibRelativePath=..\..\..\lib
!	else if "$(HierarchyLevel)" == "5"
LibRelativePath=..\..\..\..\lib
!	else
!		error Bad HierarchyLevel=$(HierarchyLevel)
!	endif
!endif

!ifndef LBD
LBD=$(ProjDir)\$(LibRelativePath)
!endif

!ifndef COPYPDB
COPYPDB=1
!endif

targets=

!if !defined( DontCopyLib ) && exist( "$(SRD)\$(TargetName).lib" )
targets=$(targets) "$(LBD)" "$(LBD)\$(TargetName).lib"
doCopyLibrary=1
!endif

!if defined(Platform) && "$(Platform)" == "x64" && !defined( DontCopyLib ) && exist( "$(SRD)\$(TargetName)64.lib" )
targets=$(targets) "$(LBD)" "$(LBD)\$(TargetName)64.lib"
doCopyLibrary=1
!endif

!if !defined( DontCopyLib ) && exist( "$(SRD)\$(TargetName).Win32.Debug.lib" )
targets=$(targets) "$(LBD)" "$(LBD)\$(TargetName).Win32.Debug.lib"
doCopyLibrary=1
!endif

!if !defined( DontCopyLib ) && exist( "$(SRD)\$(TargetName).Win32.Release.lib" )
targets=$(targets) "$(LBD)" "$(LBD)\$(TargetName).Win32.Release.lib"
doCopyLibrary=1
!endif

!if !defined( DontCopyLib ) && exist( "$(SRD)\$(TargetName).Win32.Final.lib" )
targets=$(targets) "$(LBD)" "$(LBD)\$(TargetName).Win32.Final.lib"
doCopyLibrary=1
!endif

!if !defined( DontCopyLib ) && exist( "$(SRD)\$(TargetName).x64.Debug.lib" )
targets=$(targets) "$(LBD)" "$(LBD)\$(TargetName).x64.Debug.lib"
doCopyLibrary=1
!endif

!if !defined( DontCopyLib ) && exist( "$(SRD)\$(TargetName).x64.Release.lib" )
targets=$(targets) "$(LBD)" "$(LBD)\$(TargetName).x64.Release.lib"
doCopyLibrary=1
!endif

!if !defined( DontCopyLib ) && exist( "$(SRD)\$(TargetName).x64.Final.lib" )
targets=$(targets) "$(LBD)" "$(LBD)\$(TargetName).x64.Final.lib"
doCopyLibrary=1
!endif

!if exist( "$(SRD)\$(TargetName).dll" ) 
targets=$(targets) "$(BRD)" "$(BRD)\$(TargetName).dll"
doCopyExecutable=1
!endif

!if exist( "$(SRD)\$(TargetName).exe" ) 
targets=$(targets) "$(BRD)" "$(BRD)\$(TargetName).exe"
doCopyExecutable=1
!endif

!if exist( "$(SRD)\$(TargetName).drv" ) 
targets=$(targets) "$(BRD)" "$(BRD)\$(TargetName).drv"
doCopyExecutable=1
!endif

!if exist( "$(SRD)\$(TargetName).map" )
targets=$(targets) "$(BRD)" "$(BRD)\$(TargetName).map"
!endif

!if exist( "$(SRD)\$(TargetName).pdb" )
!	if exist( "$(BRD)\$(TargetName).pdb" )
targets=$(targets) "$(BRD)" "$(BRD)\$(TargetName).pdb"
!	endif
!	if exist( "$(LBD)\$(TargetName).pdb" )
targets=$(targets) "$(LBD)" "$(LBD)\$(TargetName).pdb"
!	endif
!	if ("$(COPYPDB)" != "") && ("$(COPYPDB)" != "0")
!		ifdef doCopyExecutable
targets=$(targets) "$(BRD)" "$(BRD)\$(TargetName).pdb"
!		elseifdef doCopyLibrary
targets=$(targets) "$(LBD)" "$(LBD)\$(TargetName).pdb"
!		endif
!	endif
!endif

!undef signCommand

!if defined( AUTHENTICODE_CERT ) && defined( AUTHENTICODE_PASS )
signCommand=@AbbyySignTool.exe sign /f "$(ROOT)\$(AUTHENTICODE_CERT)" /p "$(AUTHENTICODE_PASS)"
!elseif defined( AUTHENTICODE_CERT_HASH ) 
signCommand=@AbbyySignTool.exe sign /sm /a /sha1 "%AUTHENTICODE_CERT_HASH%"
!endif

!if defined( signCommand ) && defined( AUTHENTICODE_TIME_SERVER )
signCommand=$(signCommand) /t "$(AUTHENTICODE_TIME_SERVER)"
!endif

# create build results directory
"$(BRD)" :
	-1 md "$(BRD)"

# create library directory
"$(LBD)" :
	-1 md "$(LBD)"

#Copy dependencies
"$(LBD)\$(TargetName).lib" : "$(SRD)\$(TargetName).lib"
	copy $? $@

"$(LBD)\$(TargetName)64.lib" : "$(SRD)\$(TargetName)64.lib"
	copy $? $@

"$(LBD)\$(TargetName).Win32.Debug.lib" : "$(SRD)\$(TargetName).Win32.Debug.lib"
	copy $? $@

"$(LBD)\$(TargetName).Win32.Release.lib" : "$(SRD)\$(TargetName).Win32.Release.lib"
	copy $? $@

"$(LBD)\$(TargetName).Win32.Final.lib" : "$(SRD)\$(TargetName).Win32.Final.lib"
	copy $? $@

"$(LBD)\$(TargetName).x64.Debug.lib" : "$(SRD)\$(TargetName).x64.Debug.lib"
	copy $? $@

"$(LBD)\$(TargetName).x64.Release.lib" : "$(SRD)\$(TargetName).x64.Release.lib"
	copy $? $@

"$(LBD)\$(TargetName).x64.Final.lib" : "$(SRD)\$(TargetName).x64.Final.lib"
	copy $? $@

"$(BRD)\$(TargetName).dll" : "$(SRD)\$(TargetName).dll"
!if defined( signCommand )
	@echo Digitally signing $?
	$(signCommand) $?
!endif
	copy $? $@

"$(BRD)\$(TargetName).exe" : "$(SRD)\$(TargetName).exe"
!if defined( signCommand )
	@echo Digitally signing $?
	$(signCommand) $?
!endif
	copy $? $@

"$(BRD)\$(TargetName).drv" : "$(SRD)\$(TargetName).drv"
!if defined( signCommand )
	@echo Digitally signing $?
	$(signCommand) $?
!endif
	copy $? $@

"$(BRD)\$(TargetName).map" : "$(SRD)\$(TargetName).map"
	copy $? $@

"$(BRD)\$(TargetName).pdb" : "$(SRD)\$(TargetName).pdb"
	copy $? $@

"$(LBD)\$(TargetName).pdb" : "$(SRD)\$(TargetName).pdb"
	copy $? $@

# Additional dependencies
!ifdef CustomCopy
!include "$(CustomCopy)"
!endif

# next pseudotarget have to be placed after !include "$(CustomCopy)" command
copyAllFiles: $(targets)
	@echo $(SRD)  $(TargetName) >"$(SRD)\copyTimeStamp.txt"

debugPrint:
	@echo Current directory:
	@cd
	@echo ROOT=$(ROOT)
	@echo TargetDir=$(TargetDir)
	@echo TargetName=$(TargetName)
	@echo ProjDir=$(ProjDir)
	@echo SRD=$(SRD)
	@echo BRD=$(BRD)
	@echo LBD=$(LBD)
!ifdef DontCopyLib
	@echo DontCopyLib defined
!endif
	@echo Targets:
	@echo $(targets)
