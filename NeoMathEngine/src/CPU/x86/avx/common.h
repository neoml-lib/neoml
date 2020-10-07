/* Copyright © 2017-2020 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

#pragma once

#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>

#include <NeoMathEngine/Platforms.h>
#include <NeoMathEngine/OpenMP.h>

#include <immintrin.h>

#if FINE_PLATFORM( FINE_WINDOWS )
#define FME_DLL_EXPORT __declspec( dllexport )
#elif FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
#define FME_DLL_EXPORT __attribute__((visibility("default")))
#else
#error "Platform isn't supported!"

#endif
using namespace std;
