@echo off
setlocal EnableDelayedExpansion

set ARCH=x64
set ENABLE_TEST=ON

if "%1"=="" (
    echo Using x64 arch by default
) else (
    if "%1"=="Win32" (
        set ARCH=Win32
    ) else ( 
        if "%1"=="-notest" (
            set ENABLE_TEST=OFF
        ) else (
            if not "%1"== "x64" (
                echo Unknown arch %1. Use Win32/x64!
                exit /b 1
            )
        )
    )
)

if "%2"=="-notest" (
    set ENABLE_TEST=OFF
)

if not defined NeoML_BUILD_DIR (
    set NeoML_BUILD_DIR=%ROOT%\_cmake_working_dir\NeoML
)

if exist %NeoML_BUILD_DIR%\%ARCH% (
    rmdir /S /Q %NeoML_BUILD_DIR%\%ARCH%
)
mkdir %NeoML_BUILD_DIR%\%ARCH% || exit /b !ERRORLEVEL!

echo Generate %ARCH%

if %ARCH% == Win32 (
    cmake -G "Visual Studio 14 2015" -A %ARCH% -DUSE_FINE_OBJECTS=ON -DNeoML_BUILD_TESTS=%ENABLE_TEST% -DNeoMathEngine_BUILD_TESTS=%ENABLE_TEST% -DCMAKE_SYSTEM_VERSION=8.1 -B %NeoML_BUILD_DIR%/%ARCH% %ROOT%/NeoML/NeoML || exit /b !ERRORLEVEL!
) else (
    cmake -G "Visual Studio 14 2015" -A %ARCH% -T cuda=%ROOT%/ThirdParty/CUDA/Windows -DUSE_FINE_OBJECTS=ON -DNeoML_BUILD_TESTS=%ENABLE_TEST% -DNeoMathEngine_BUILD_TESTS=%ENABLE_TEST% -DCMAKE_SYSTEM_VERSION=8.1 -B %NeoML_BUILD_DIR%/%ARCH% %ROOT%/NeoML/NeoML || exit /b !ERRORLEVEL!
)
