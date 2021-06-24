@echo off
SETLOCAL EnableDelayedExpansion

call %ROOT%\FineObjects\Cmake\build_android.cmd || exit /b !ERRORLEVEL!

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

