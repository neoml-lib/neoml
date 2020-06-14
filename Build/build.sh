#!/bin/bash
set -e

function printError() {
    echo -e "\033[31m$1\033[0m"
}

if [[ ! $ROOT ]]; then
    printError "ROOT is not set!"
    exit 1
fi

# Get hostname (Linux or Darwin)
HOSTNAME="$(uname)"
if [[ ! "$HOSTNAME" =~ ^(Linux|Darwin)$ ]]; then
    printError "Unknown hostname: $HOSTNAME!"
    exit 1
fi

# Build target (Linux/Android/IOS/Darwin)
if [[ ! $FINE_CMAKE_BUILD_TARGET ]]; then
    FINE_CMAKE_BUILD_TARGET=$1
elif [[ ! -z "$1" ]]; then
    echo "Variable FINE_CMAKE_BUILD_TARGET already set! Passing \"$1\" argument has no effect!"
fi

# Build config. Map FINE_CMAKE_BUILD_CONFIG -> CMAKE_BUILD_CONFIG
if [[ $FINE_CMAKE_BUILD_CONFIG == "Debug" ]]; then
    CMAKE_BUILD_CONFIG=Debug
elif [[ $FINE_CMAKE_BUILD_CONFIG == "Release" ]]; then
    CMAKE_BUILD_CONFIG=RelWithDebInfo
elif [[ $FINE_CMAKE_BUILD_CONFIG == "Final" ]]; then
    CMAKE_BUILD_CONFIG=Release
else
    printError "FINE_CMAKE_BUILD_CONFIG is not Debug/Release/Final!"
    exit 1
fi

# Set FINE_CMAKE_BUILD_THREAD_COUNT=1 by default
if [[ ! $FINE_CMAKE_BUILD_THREAD_COUNT ]]; then
    NUMBER_CORES=$(getconf _NPROCESSORS_ONLN)
    FINE_CMAKE_BUILD_THREAD_COUNT=$NUMBER_CORES
    echo "Set FINE_CMAKE_BUILD_THREAD_COUNT=${FINE_CMAKE_BUILD_THREAD_COUNT} by default."
fi


# Target specific settings
if [[ $FINE_CMAKE_BUILD_TARGET =~ ^(Linux|Darwin)$ ]]; then
    # Check hostname
    if [[ $HOSTNAME != $FINE_CMAKE_BUILD_TARGET ]]; then
        printError "HOSTNAME is not $FINE_CMAKE_BUILD_TARGET!"
        exit 1
    fi
    
    export FINE_CMAKE_BUILD_ARCH=x86_64
    
elif [[ $FINE_CMAKE_BUILD_TARGET == "IOS" ]]; then
    if [[ $HOSTNAME != Darwin ]]; then
        printError "HOSTNAME must be Darwin!"
        exit 1
    fi
    
    if [[ ! "$FINE_CMAKE_BUILD_ARCH" =~ ^(x86_64|arm64)$ ]]; then
        printError "FINE_CMAKE_BUILD_ARCH is not x86_64/arm64"
        exit 1
    fi
        
    ADD_ARGS="-GXcode -DIOS_ARCH=$FINE_CMAKE_BUILD_ARCH"
        
elif [[ $FINE_CMAKE_BUILD_TARGET == "Android" ]]; then
    if [[ ! "$FINE_CMAKE_APP_ABI" =~ ^(armeabi-v7a|arm64-v8a|x86|x86_64)$ ]]; then
        printError "FINE_CMAKE_APP_ABI is not armeabi-v7a/arm64-v8a/x86/x86_64!"
        exit 1
    fi
    
    FINE_CMAKE_BUILD_ARCH=${FINE_CMAKE_APP_ABI}
    
    # NDK Root
    if [[ ! $FINE_CMAKE_NDK_ROOT ]]; then
        printError "FINE_CMAKE_NDK_ROOT is not set!"
        exit 1
    fi
    
    ADD_ARGS="-DANDROID_ABI=${FINE_CMAKE_APP_ABI} -DFINE_CMAKE_NDK_ROOT:PATH=${FINE_CMAKE_NDK_ROOT}"
else
	printError "FINE_CMAKE_BUILD_TARGET is not Linux/Darwin/IOS/Android!"
	exit 1
fi

CMAKE_WORKING_DIR=$ROOT/_cmake_working_dir/FineObj.${FINE_CMAKE_BUILD_TARGET}.${FINE_CMAKE_BUILD_CONFIG}.${FINE_CMAKE_BUILD_ARCH}

[ -d ${CMAKE_WORKING_DIR} ] || mkdir -p ${CMAKE_WORKING_DIR}
pushd ${CMAKE_WORKING_DIR}

# Build FineObjects
cmake \
    -DFINE_CMAKE_BUILD_TARGET:STRING=${FINE_CMAKE_BUILD_TARGET} \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=${FINE_CMAKE_VERBOSE} \
    -DFINE_CMAKE_BUILD_CONFIG:STRING=${FINE_CMAKE_BUILD_CONFIG} \
    -DFINE_CMAKE_BUILD_ARCH:STRING=${FINE_CMAKE_BUILD_ARCH} \
    -DROOT:PATH=${ROOT} \
    -DCMAKE_MODULE_PATH:PATH=${ROOT}/FineObjects/Cmake \
    ${ADD_ARGS} \
    ${ROOT}

cmake --build . --target install --config ${CMAKE_BUILD_CONFIG} --parallel ${FINE_CMAKE_BUILD_THREAD_COUNT}
	
popd

# Build FineML
CMAKE_WORKING_DIR=$ROOT/_cmake_working_dir/NeoML.${FINE_CMAKE_BUILD_TARGET}.${FINE_CMAKE_BUILD_CONFIG}.${FINE_CMAKE_BUILD_ARCH}
[ -d ${CMAKE_WORKING_DIR} ] || mkdir -p ${CMAKE_WORKING_DIR}
pushd ${CMAKE_WORKING_DIR}

if [[ $FINE_CMAKE_BUILD_TARGET == "IOS" ]]; then
	cmake -G Xcode -DUSE_FINE_OBJECTS=ON -DCMAKE_TOOLCHAIN_FILE=${ROOT}/NeoML/NeoML/cmake/ios.toolchain.cmake -DIOS_ARCH=${FINE_CMAKE_BUILD_ARCH} ${ROOT}/NeoML/NeoML
elif [[ $FINE_CMAKE_BUILD_TARGET == "Linux" && $FINE_CMAKE_BUILD_ARCH == "x86" ]]; then
	cmake -DUSE_FINE_OBJECTS=ON -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_CONFIG} -DCMAKE_CXX_FLAGS=-m32 -DCMAKE_C_FLAGS=-m32 ${ROOT}/NeoML/NeoML
elif [[ $FINE_CMAKE_BUILD_TARGET =~ ^(Linux|Darwin)$ ]]; then
	cmake -DUSE_FINE_OBJECTS=ON -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_CONFIG} ${ROOT}/NeoML/NeoML
elif [[ $FINE_CMAKE_BUILD_TARGET = "Android" ]]; then
	cmake -DUSE_FINE_OBJECTS=ON -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_CONFIG} -DCMAKE_TOOLCHAIN_FILE=${FINE_CMAKE_NDK_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=${FINE_CMAKE_APP_ABI} ${ROOT}/NeoML/NeoML
fi

cmake --build . --target install --config ${CMAKE_BUILD_CONFIG} --parallel ${FINE_CMAKE_BUILD_THREAD_COUNT}
	
popd





