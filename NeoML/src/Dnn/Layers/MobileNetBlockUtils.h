/* Copyright © 2017-2023 ABBYY Production LLC

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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

inline CPtr<CDnnBlob> MobileNetParam( const CPtr<CDnnBlob>& blob )
{
	return blob == nullptr ? nullptr : blob->GetCopy();
}

inline CPtr<CDnnBlob> MobileNetFreeTerm( CDnnBlob* freeTerm )
{
	if( freeTerm == nullptr ) {
		return nullptr;
	}

	CDnnBlobBuffer<> buffer( *freeTerm, TDnnBlobBufferAccess::Read );
	for( int i = 0; i < buffer.Size(); ++i ) {
		if( buffer[i] != 0 ) {
			return freeTerm;
		}
	}

	return nullptr;
}

inline bool IsValidMobileNetBlockActivation( const CActivationDesc& desc )
{
	if( desc.GetType() == AF_ReLU || desc.GetType() == AF_HSwish ) {
		return true;
	}

	if( desc.GetType() == AF_Linear ) {
		if( !desc.HasParam() ) {
			return false;
		}
		CLinearLayer::CParam param = desc.GetParam<CLinearLayer::CParam>();
		return param.Multiplier == 1.f && param.FreeTerm == 0.f;
	}

	return false;
}

inline float MobileNetReluParam( const CActivationDesc& desc )
{
	return desc.GetType() == AF_ReLU ? desc.GetParam<CReLUActivationParam>().UpperThreshold : 0.f;
}

} // namespace NeoML
