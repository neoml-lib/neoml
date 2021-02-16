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

#include <NeoML/Dnn/Layers/TransposeLayer.h>

namespace NeoML {

CTransposeLayer::CTransposeLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnTransposeLayer", false ),
	d1(TBlobDim(0)),
	d2(TBlobDim(0))
{
}

static const int TransposeLayerVersion = 2000;

void CTransposeLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( TransposeLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.SerializeEnum(d1);
	archive.SerializeEnum(d2);
}

void CTransposeLayer::Reshape()
{
	CheckInput1();
	if(d1 != d2) {
		outputDescs[0] = inputDescs[0];
		int size1 = outputDescs[0].DimSize(d1);
		int size2 = outputDescs[0].DimSize(d2);
		outputDescs[0].SetDimSize(d1, size2);
		outputDescs[0].SetDimSize(d2, size1);
	} else {
		outputDescs[0] = inputDescs[0];
	}
}

void CTransposeLayer::RunOnce()
{
	outputBlobs[0]->TransposeFrom(inputBlobs[0], d1, d2);
}

void CTransposeLayer::BackwardOnce()
{
	inputDiffBlobs[0]->TransposeFrom(outputDiffBlobs[0], d1, d2);
}

CLayerWrapper<CTransposeLayer> Transpose( TBlobDim d1, TBlobDim d2 )
{
	return CLayerWrapper<CTransposeLayer>( "Transpose", [=]( CTransposeLayer* result ) {
		result->SetTransposedDimensions( d1, d2 );
	} );
}

} // namespace NeoML
