@echo off
setlocal EnableDelayedExpansion

set ARCH=x64
set ENABLE_TEST=ON
set DIR=

:parseArgs

if "%~1" == "-notest" (
	set ENABLE_TEST=OFF
	shift /1
	goto parseArgs
)

if "%~1" == "-dir" (
	if "%~2" == "" (
		echo Path must follow the -dir.
		exit /b 1
	)
	set "DIR=%~f2"
	shift /1
	shift /1
	goto parseArgs
)

if "%~1" == "Win32" (
	set ARCH=Win32
	shift /1
	goto parseArgs
)

if "%~1" == "x64" (
	set ARCH=x64
	shift /1
	goto parseArgs
)

if not "%~1" == "" (
	echo Unknown arch "%~1". Use Win32/x64!
	exit /b 1
)

if not defined NeoML_BUILD_DIR (
	set "NeoML_BUILD_DIR=%ROOT%\_cmake_working_dir\NeoML"
)

if not defined DIR (
	set "DIR=%NeoML_BUILD_DIR%\%ARCH%"
)

if not defined CMAKE_GENERATOR (
	set "CMAKE_GENERATOR=Visual Studio 16 2019"
)

if not defined CMAKE_GENERATOR_TOOLSET (
	set "CMAKE_GENERATOR_TOOLSET=v142,version=14.28,host=x64"
)
if "%CMAKE_GENERATOR_TOOLSET:cuda=%" == "%CMAKE_GENERATOR_TOOLSET%" (
	set "CMAKE_GENERATOR_TOOLSET=%CMAKE_GENERATOR_TOOLSET%,cuda=%ROOT%/ThirdParty/CUDA/Windows"
)

if not defined CMAKE_SYSTEM_VERSION (
	set "CMAKE_SYSTEM_VERSION=10"
)

if exist "%DIR%" (
    rmdir /S /Q "%DIR%"
)
mkdir "%DIR%" || exit /b !ERRORLEVEL!

echo Generating project:
echo   Architecture = "%ARCH%"
echo   Tests = "%ENABLE_TEST%"
echo   Directory = "%DIR%"
echo   Generator = "%CMAKE_GENERATOR%"
echo   Toolset = "%CMAKE_GENERATOR_TOOLSET%"
echo   Target version = "%CMAKE_SYSTEM_VERSION%"
echo.

cmake -A %ARCH% -DUSE_FINE_OBJECTS=ON -DNeoML_BUILD_TESTS=%ENABLE_TEST% -DNeoMathEngine_BUILD_TESTS=%ENABLE_TEST% -DCMAKE_SYSTEM_VERSION="%CMAKE_SYSTEM_VERSION%" -B "%DIR%" "%ROOT%/NeoML/NeoML" || exit /b !ERRORLEVEL!
