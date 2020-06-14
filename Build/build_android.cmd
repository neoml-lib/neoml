@echo off
SETLOCAL EnableDelayedExpansion

rem Check ROOT variable
if not defined ROOT (
    echo ROOT variable is not set
    exit /b 1
)

rem Check FINE_CMAKE_BUILD_CONFIG variable
if not defined FINE_CMAKE_BUILD_CONFIG (
    echo FINE_CMAKE_BUILD_CONFIG variable is not set
    exit /b 1
)

if %FINE_CMAKE_BUILD_CONFIG% == Release (
    set CMAKE_BUILD_CONFIG=RelWithDebInfo
    goto BUILD_CONFIG_OK
) else (
    if %FINE_CMAKE_BUILD_CONFIG% == Final (
        set CMAKE_BUILD_CONFIG=Release
        goto BUILD_CONFIG_OK
    ) else (
        if %FINE_CMAKE_BUILD_CONFIG% == Debug (
            set CMAKE_BUILD_CONFIG=Debug
            goto BUILD_CONFIG_OK
        )
    )
)

:BUILD_CONFIG_BAD
echo Bad FINE_CMAKE_BUILD_CONFIG value. Possible values: Final/Release/Debug
exit /b 1

:BUILD_CONFIG_OK
rem Check FINE_CMAKE_NDK_ROOT variable
if not defined FINE_CMAKE_NDK_ROOT (
    echo FINE_CMAKE_NDK_ROOT variable is not set
    exit /b 1
)

rem Check FINE_CMAKE_APP_ABI variable
if not defined FINE_CMAKE_APP_ABI (
    echo FINE_CMAKE_APP_ABI variable is not set
    exit /b 1
)
for %%G in ("armeabi-v7a"
            "arm64-v8a"
            "x86"
            "x86_64"
            ) DO (
            if "%FINE_CMAKE_APP_ABI%"=="%%~G" GOTO APP_ABI_OK
)
:APP_ABI_BAD
echo Bad FINE_CMAKE_APP_ABI value. Possible values: armeabi-v7a/arm64-v8a/x86/x86_64
exit /b 1

:APP_ABI_OK
rem Set FINE_CMAKE_BUILD_THREAD_COUNT=1 by default
if not defined FINE_CMAKE_BUILD_THREAD_COUNT  (
    set FINE_CMAKE_BUILD_THREAD_COUNT=1
)

set CMAKE_WORKING_DIR=%ROOT%\_cmake_working_dir\FineObjects.Android.%FINE_CMAKE_BUILD_CONFIG%.%FINE_CMAKE_APP_ABI%
if not exist %CMAKE_WORKING_DIR% (
    mkdir %CMAKE_WORKING_DIR% || exit /b !ERRORLEVEL!
)

pushd %CMAKE_WORKING_DIR%

echo Building FineObjects
echo Building arch: %FINE_CMAKE_APP_ABI%
echo Building configuration: %FINE_CMAKE_BUILD_CONFIG%

cmake -G"Unix Makefiles"                               			 ^
	  -DCMAKE_MAKE_PROGRAM=%FINE_CMAKE_NDK_ROOT%/prebuilt/windows-x86_64/bin/make.exe	^
	  -DCMAKE_MODULE_PATH:PATH=%ROOT%\FineObjects\Cmake			 ^
	  -DANDROID_ABI=%FINE_CMAKE_APP_ABI%       	      			 ^
	  -DFINE_CMAKE_NDK_ROOT:PATH=%FINE_CMAKE_NDK_ROOT% 		 	 ^
	  -DFINE_CMAKE_BUILD_CONFIG:STRING=%FINE_CMAKE_BUILD_CONFIG% ^
	  -DFINE_CMAKE_BUILD_TARGET:STRING=Android 			 		 ^
	  %ROOT% ^ || exit /b !ERRORLEVEL!

cmake --build . -j %FINE_CMAKE_BUILD_THREAD_COUNT% --target install || exit /b !ERRORLEVEL!
popd

set CMAKE_WORKING_DIR=%ROOT%\_cmake_working_dir\NeoML.Android.%FINE_CMAKE_BUILD_CONFIG%.%FINE_CMAKE_APP_ABI%

if not exist %CMAKE_WORKING_DIR% (
    mkdir %CMAKE_WORKING_DIR% || exit /b !ERRORLEVEL!
)
pushd %CMAKE_WORKING_DIR%

echo Building NeoML
echo Building arch: %FINE_CMAKE_APP_ABI%
echo Building configuration: %FINE_CMAKE_BUILD_CONFIG%

cmake -G"Unix Makefiles" -DCMAKE_MAKE_PROGRAM=%FINE_CMAKE_NDK_ROOT%/prebuilt/windows-x86_64/bin/make.exe -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_CONFIG% -DCMAKE_CXX_FLAGS="-w" -DANDROID_ABI=%FINE_CMAKE_APP_ABI% -DUSE_FINE_OBJECTS=ON -DCMAKE_TOOLCHAIN_FILE=%FINE_CMAKE_NDK_ROOT%/build/cmake/android.toolchain.cmake %ROOT%/NeoML/NeoML || exit /b !ERRORLEVEL!

cmake --build . --target install || exit /b !ERRORLEVEL!
popd

