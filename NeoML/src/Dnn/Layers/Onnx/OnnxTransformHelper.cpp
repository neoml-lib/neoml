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

#include <NeoML/Dnn/Layers/Onnx/OnnxTransformHelper.h>

namespace NeoML {

static const int OnnxTransformHelperVersion = 0;

COnnxTransformHelper::COnnxTransformHelper( IMathEngine& mathEngine ) :
	COnnxLayerBase( mathEngine, "OnnxTransformHelper" )
{
	transformInfo.Add( BD_Count, BD_Count );
}

void COnnxTransformHelper::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxTransformHelperVersion );
	COnnxLayerBase::Serialize( archive );
	transformInfo.Serialize( archive );
}

void COnnxTransformHelper::CalculateShapes()
{
	CheckArchitecture( GetInputCount() == 1, GetPath(), "Layer must have 1 input" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "Layer must have 1 output" );

	const CBlobDesc& inputDesc = inputShapeBlobs[0] == nullptr ? inputDescs[0] : inputShapeBlobs[0]->GetDesc();
	outputDesc = inputDesc;

	NeoPresume( transformInfo.Size() == BD_Count );
	for( int i = 0; i < transformInfo.Size(); ++i ) {
		NeoPresume( transformInfo[i] >= BD_BatchLength && transformInfo[i] <= BD_Count );
		if( transformInfo[i] == BD_Count ) {
			outputDesc.SetDimSize( i, 1 );
		} else {
			outputDesc.SetDimSize( i, inputDesc.DimSize( transformInfo[i] ) );
		}
	}

	if( inputShapeBlobs[0] != nullptr ) {
		outputShapeBlobs[0] = inputShapeBlobs[0]->GetCopy();
		outputShapeBlobs[0]->ReinterpretDimensions( outputDesc );
	} else {
		NeoPresume( inputDescs[0].BlobSize() == outputDesc.BlobSize() );
		outputDescs[0] = outputDesc;
		EnableInPlace( InputsMayBeOverwritten() );
	}
}

void COnnxTransformHelper::RunOnce()
{
	if( inputShapeBlobs[0] != nullptr ) {
		return;
	}

	if( inputBlobs[0]->GetDataType() == CT_Float && inputBlobs[0]->GetData() != outputBlobs[0]->GetData() ) {
		MathEngine().VectorCopy( outputBlobs[0]->GetData(), inputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize() );
	} else if( inputBlobs[0]->GetDataType() == CT_Int && inputBlobs[0]->GetData<int>() != outputBlobs[0]->GetData<int>() ) {
		MathEngine().VectorCopy( outputBlobs[0]->GetData<int>(), inputBlobs[0]->GetData<int>(), outputBlobs[0]->GetDataSize() );
	} else {
		outputBlobs[0]->ReinterpretDimensions( outputDesc );
	}
}

} // namespace NeoML
