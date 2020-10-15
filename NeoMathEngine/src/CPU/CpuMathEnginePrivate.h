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

// These functions only use raw pointers, do not contain any omp sections inside and perform no checks

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>

#if defined(NEOML_USE_SSE)

#include <CpuX86.h>
#include <CpuX86MathEngineVectorMathPrivate.h>
#include <NeoMathEngineAvxDll.h>

#elif defined(NEOML_USE_NEON)

#include <CpuArm.h>
#include <CpuArmMathEngineVectorMathPrivate.h>

#else

#error "Platform isn't supported!"

#endif
