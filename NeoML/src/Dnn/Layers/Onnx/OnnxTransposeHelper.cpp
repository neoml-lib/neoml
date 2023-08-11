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

#include <NeoML/Dnn/Layers/Onnx/OnnxTransposeHelper.h>

namespace NeoML {

// Calculates output size of Onnx transpose helper
static CBlobDesc calcOnnxTransposeHelperDesc( const CBlobDesc& inputDesc, TBlobDim firstDim, TBlobDim secondDim )
{
	CBlobDesc outputDesc = inputDesc;
	const int firstDimSize = inputDesc.DimSize( firstDim );
	const int secondDimSize = inputDesc.DimSize( secondDim );
	outputDesc.SetDimSize( firstDim, secondDimSize );
	outputDesc.SetDimSize( secondDim, firstDimSize );
	return outputDesc;
}

//---------------------------------------------------------------------------------------------------------------------

static const int OnnxTransposeHelperVersion = 0;

COnnxTransposeHelper::COnnxTransposeHelper( IMathEngine& mathEngine ) :
	COnnxLayerBase( mathEngine, "OnnxTransposeHelper" )
{
	dims[0] = BD_Count;
	dims[1] = BD_Count;
}

COnnxTransposeHelper::COnnxTransposeHelper( IMathEngine& mathEngine, const CFastArray<TBlobDim, 8>& _inputLayout,
		const CFastArray<TBlobDim, 8>& _outputLayout ) :
	COnnxTransposeHelper( mathEngine )
{
	_inputLayout.CopyTo( inputLayout );
	_outputLayout.CopyTo( outputLayout );
}

void COnnxTransposeHelper::SetDims( TBlobDim firstDim, TBlobDim secondDim )
{
	NeoPresume( firstDim >= BD_BatchLength && firstDim < BD_Count );
	NeoPresume( secondDim >= BD_BatchLength && secondDim < BD_Count );
	NeoPresume( firstDim != secondDim );
	dims[0] = firstDim;
	dims[1] = secondDim;
}

void COnnxTransposeHelper::GetDims( TBlobDim& firstDim, TBlobDim& secondDim ) const
{
	firstDim = dims[0];
	secondDim = dims[1];
}

void COnnxTransposeHelper::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxTransposeHelperVersion );
	COnnxLayerBase::Serialize( archive );
	archive.SerializeEnum( dims[0] );
	archive.SerializeEnum( dims[1] );
}

void COnnxTransposeHelper::CalculateShapes()
{
	if( inputShapeBlobs[0] == nullptr ) {
		outputDescs[0] = calcOnnxTransposeHelperDesc( inputDescs[0], dims[0], dims[1] );
		return;
	}

	CBlobDesc outputDesc = calcOnnxTransposeHelperDesc( inputShapeBlobs[0]->GetDesc(), dims[0], dims[1] );
	outputShapeBlobs[0] = CDnnBlob::CreateBlob( inputShapeBlobs[0]->GetMathEngine(), outputDesc.GetDataType(), outputDesc );
	outputShapeBlobs[0]->TransposeFrom( inputShapeBlobs[0], dims[0], dims[1] );
}

void COnnxTransposeHelper::RunOnce()
{
	if( inputShapeBlobs[0] == nullptr ) {
		outputBlobs[0]->TransposeFrom( inputBlobs[0], dims[0], dims[1] );
	}
}

} // namespace NeoML
