/* Copyright © 2017-2021 ABBYY Production LLC

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

#include <common.h>
#pragma hdrstop

#include "PyMathEngine.h"

CPyMathEngine::CPyMathEngine( CPyMathEngineOwner& owner ) :
	mathEngineOwner( &owner )
{
}

CPyMathEngine::CPyMathEngine( const std::string& type, int threadCount, int index )
{
	if( type == "cpu" ) {
		mathEngineOwner = new CPyMathEngineOwner( CreateCpuMathEngine( threadCount, 0 ) );
	} else if( type == "gpu" ) {
		mathEngineOwner = new CPyMathEngineOwner( CreateGpuMathEngine( index, 0 ) );
	} else {
		assert( false );
	}
}

std::string CPyMathEngine::GetInfo() const
{
	CMathEngineInfo gpuInfo;
	mathEngineOwner->MathEngine().GetMathEngineInfo( gpuInfo );

	std::string info;
	switch( gpuInfo.Type ) {
		case MET_Cuda:
			info = "CUDA: ";
			break;
		case MET_Vulkan:
			info = "Vulkan: ";
			break;
		default:
			break;
	};

	info += gpuInfo.Name;

	return info;
}

long long CPyMathEngine::GetPeakMemoryUsage()
{
	return mathEngineOwner->MathEngine().GetPeakMemoryUsage();
}

void CPyMathEngine::CleanUp()
{
	return mathEngineOwner->MathEngine().CleanUp();
}

void CPyMathEngine::VectorAdd(const CPyBlob& first, const CPyBlob& second, const CPyBlob& result)
{
	if( first.Blob()->GetDataType() == CT_Float ) {
		mathEngineOwner->MathEngine().VectorAdd(first.Blob()->GetData<float>(), second.Blob()->GetData<float>(),
			result.Blob()->GetData<float>(), first.Blob()->GetDataSize());
	} else {
		mathEngineOwner->MathEngine().VectorAdd(first.Blob()->GetData<int>(), second.Blob()->GetData<int>(),
			result.Blob()->GetData<int>(), first.Blob()->GetDataSize());
	}
}

void CPyMathEngine::VectorSub(const CPyBlob& first, const CPyBlob& second, const CPyBlob& result)
{
	if( first.Blob()->GetDataType() == CT_Float ) {
		mathEngineOwner->MathEngine().VectorAdd(first.Blob()->GetData<float>(), second.Blob()->GetData<float>(),
			result.Blob()->GetData<float>(), first.Blob()->GetDataSize());
	} else {
		mathEngineOwner->MathEngine().VectorAdd(first.Blob()->GetData<int>(), second.Blob()->GetData<int>(),
			result.Blob()->GetData<int>(), first.Blob()->GetDataSize());
	}
}

void CPyMathEngine::VectorEltwiseMultiply(const CPyBlob& first, const CPyBlob& result, const CPyBlob& multiplier)
{
	mathEngineOwner->MathEngine().VectorEltwiseMultiply(first.Blob()->GetData<float>(), result.Blob()->GetData<float>(),
		first.Blob()->GetDataSize(), multiplier.Blob()->GetData<float>() );
}

//------------------------------------------------------------------------------------------------------------

void InitializeMathEngine(py::module& m)
{
	py::class_<CPyMathEngine>(m, "MathEngine")
		.def( py::init([]( const CPyMathEngine& mathEngine ) { return new CPyMathEngine( mathEngine.MathEngineOwner() ); }) )
		.def( py::init([]( const std::string& type, int threadCount, int index ) { return new CPyMathEngine( type, threadCount, index ); }) )
		.def( "get_info", &CPyMathEngine::GetInfo, py::return_value_policy::reference )
		.def( "get_peak_memory_usage", &CPyMathEngine::GetPeakMemoryUsage, py::return_value_policy::reference )
		.def( "clean_up", &CPyMathEngine::CleanUp, py::return_value_policy::reference )
		.def( "add", &CPyMathEngine::VectorAdd, py::return_value_policy::reference )
		.def( "sub", &CPyMathEngine::VectorSub, py::return_value_policy::reference )
	;

	m.def("enum_gpu", []() {
		std::unique_ptr<IGpuMathEngineManager> manager( CreateGpuMathEngineManager() );
		CMathEngineInfo info;

		auto t = py::tuple(manager->GetMathEngineCount());
		for( int i = 0; i < t.size(); ++i ) {
			manager->GetMathEngineInfo( i, info );

			std::string name;
			switch( info.Type ) {
				case MET_Cuda:
					name = "CUDA: ";
					break;
				case MET_Vulkan:
					name = "Vulkan: ";
					break;
				default:
					break;
			};

			t[i] = name + std::string( info.Name );
		}
		return t;
	});
}
