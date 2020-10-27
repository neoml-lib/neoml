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

#include <common.h>
#pragma hdrstop

#include <DnnConv.h>
#include <AvxDll.h>

namespace NeoML {

class CAvxDllLoader : public IAvxDllLoader {
public:
	std::unique_ptr<ISimdConvolutionEngine> InitAvxConvolutionEngine( int filterCount, int channelCount, int filterHeight, int filterWidth ) override;
};

std::unique_ptr<ISimdConvolutionEngine> CAvxDllLoader::InitAvxConvolutionEngine( int filterCount, int channelCount, int filterHeight, int filterWidth )
{
	if( CBlobConvolutionFabric::IsBlobConvolutionAvailable( filterCount, channelCount, filterHeight, filterWidth ) ) {
		return CBlobConvolutionFabric::GetProperInstance( filterCount, channelCount, filterHeight, filterWidth );
	} else {
		return nullptr;
	}
}

extern "C"
FME_DLL_EXPORT
IAvxDllLoader* GetAvxDllLoaderInstance()
{
	static CAvxDllLoader avxDllLoaderInstance;
	return &avxDllLoaderInstance;
}

}
