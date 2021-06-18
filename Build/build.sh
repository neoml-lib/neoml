#!/bin/bash
set -e

source ${ROOT}/FineObjects/Cmake/build.sh $1

# Build FineML
CMAKE_WORKING_DIR=$ROOT/_cmake_working_dir/NeoML.${FINE_CMAKE_BUILD_TARGET}.${FINE_CMAKE_BUILD_CONFIG}.${FINE_CMAKE_BUILD_ARCH}
[ -d ${CMAKE_WORKING_DIR} ] || mkdir -p ${CMAKE_WORKING_DIR}
pushd ${CMAKE_WORKING_DIR}

if [[ $FINE_CMAKE_BUILD_TARGET == "IOS" ]]; then
	cmake -G Xcode -DUSE_FINE_OBJECTS=ON -DCMAKE_TOOLCHAIN_FILE=${ROOT}/NeoML/cmake/ios.toolchain.cmake -DIOS_ARCH=${FINE_CMAKE_BUILD_ARCH} ${ROOT}/NeoML/NeoML
elif [[ $FINE_CMAKE_BUILD_TARGET == "Linux" && $FINE_CMAKE_BUILD_ARCH == "x86" ]]; then
	cmake -DUSE_FINE_OBJECTS=ON -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_CONFIG} -DCMAKE_CXX_FLAGS=-m32 -DCMAKE_C_FLAGS=-m32 ${ROOT}/NeoML/NeoML
elif [[ $FINE_CMAKE_BUILD_TARGET =~ ^(Linux|Darwin)$ ]]; then
	cmake -DUSE_FINE_OBJECTS=ON -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_CONFIG} ${ROOT}/NeoML/NeoML
elif [[ $FINE_CMAKE_BUILD_TARGET = "Android" ]]; then
	cmake -DUSE_FINE_OBJECTS=ON -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_CONFIG} -DCMAKE_TOOLCHAIN_FILE=${FINE_CMAKE_NDK_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_ABI=${FINE_CMAKE_APP_ABI} ${ROOT}/NeoML/NeoML
fi

cmake --build . --target install --config ${CMAKE_BUILD_CONFIG} --parallel ${FINE_CMAKE_BUILD_THREAD_COUNT}
	
popd





