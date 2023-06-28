/* Copyright © 2017-2023 ABBYY

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

#include <NeoML/Dnn/Rowwise/Activation.h>

namespace NeoML {

CRowwiseOperationDesc* CActivationRowwise::GetDesc( const CBlobDesc& )
{
	switch( desc.GetType() ) {
		case AF_HSwish:
		case AF_Sigmoid:
			return mathEngine.InitRowwiseActivation( desc.GetType(), 0, 0);
		case AF_ReLU:
			return mathEngine.InitRowwiseActivation( AF_ReLU,
				desc.HasParam() ? desc.GetParam<CReLULayer::CParam>().UpperThreshold : 0,
				0 );
	}
	NeoAssert( false );
	return nullptr;
}

void CActivationRowwise::Serialize( CArchive& archive )
{
	(void) archive.SerializeVersion( 0 ); // version
	if( archive.IsStoring() ) {
		StoreActivationDesc( desc, archive );
	} else {
		desc = LoadActivationDesc( archive );
	}
}

REGISTER_NEOML_ROWWISE_OPERATION( CActivationRowwise, "ActivationRowwiseOperation" )

} // namespace NeoML
