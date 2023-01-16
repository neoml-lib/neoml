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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/Onnx/TransposeReshaper.h>

namespace NeoML {

static CBlobDesc getTransposedDesc( const CBlobDesc& inputDesc, TBlobDim firstDim, TBlobDim secondDim )
{
	CBlobDesc outputDesc = inputDesc;
	const int firstDimSize = inputDesc.DimSize( firstDim );
	const int secondDimSize = inputDesc.DimSize( secondDim );
	outputDesc.SetDimSize( firstDim, secondDimSize );
	outputDesc.SetDimSize( secondDim, firstDimSize );
	return outputDesc;
}

//---------------------------------------------------------------------------------------------------------------------

static const int TransposeReshaperVersion = 0;

CTransposeReshaper::CTransposeReshaper( IMathEngine& mathEngine ) :
	COnnxLayerBase( mathEngine, "TransposeReshaper" )
{
	dims[0] = BD_Count;
	dims[1] = BD_Count;
}

void CTransposeReshaper::SetDims( TBlobDim firstDim, TBlobDim secondDim )
{
	NeoPresume( firstDim >= BD_BatchLength && firstDim < BD_Count );
	NeoPresume( secondDim >= BD_BatchLength && secondDim < BD_Count );
	NeoPresume( firstDim != secondDim );
	dims[0] = firstDim;
	dims[1] = secondDim;
}

void CTransposeReshaper::GetDims( TBlobDim& firstDim, TBlobDim& secondDim ) const
{
	firstDim = dims[0];
	secondDim = dims[1];
}

void CTransposeReshaper::Serialize( CArchive& archive )
{
	archive.SerializeVersion( TransposeReshaperVersion );
	COnnxLayerBase::Serialize( archive );
	archive.SerializeEnum( dims[0] );
	archive.SerializeEnum( dims[1] );
}

void CTransposeReshaper::CalculateShapes()
{
	if( inputShapeBlobs[0] == nullptr ) {
		outputDescs[0] = getTransposedDesc( inputDescs[0], dims[0], dims[1] );
		return;
	}

	CBlobDesc outputDesc = getTransposedDesc( inputShapeBlobs[0]->GetDesc(), dims[0], dims[1] );
	outputShapeBlobs[0] = CDnnBlob::CreateBlob( inputShapeBlobs[0]->GetMathEngine(), outputDesc );
	outputShapeBlobs[0]->TransposeFrom( inputShapeBlobs[0], dims[0], dims[1] );
}

void CTransposeReshaper::RunOnce()
{
	if( inputShapeBlobs[0] == nullptr ) {
		outputBlobs[0]->TransposeFrom( inputBlobs[0], dims[0], dims[1] );
	}
}

} // namespace NeoML
