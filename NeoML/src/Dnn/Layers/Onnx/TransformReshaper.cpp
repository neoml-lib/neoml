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

#include <NeoML/Dnn/Layers/Onnx/TransformReshaper.h>

namespace NeoML {

static const int TransformReshaperVersion = 0;

CTransformReshaper::CTransformReshaper( IMathEngine& mathEngine ) :
	CBaseReshaper( mathEngine, "TransformReshaper" )
{
	transformInfo.Add( BD_Count, BD_Count );
}

void CTransformReshaper::Serialize( CArchive& archive )
{
	archive.SerializeVersion( TransformReshaperVersion );
	CBaseReshaper::Serialize( archive );
	transformInfo.Serialize( archive );
}

void CTransformReshaper::CalculateShapes()
{
	CheckArchitecture( GetInputCount() == 1, GetPath(), "Layer must have 1 input" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "Layer must have 1 output" );
	CheckArchitecture( inputShapeBlobs[0] != nullptr, GetPath(), "Input shape blob missing" );

	const CBlobDesc inputDesc = inputShapeBlobs[0]->GetDesc();
	CBlobDesc outputDesc = inputDesc;

	NeoPresume( transformInfo.Size() == BD_Count );
	for( int i = 0; i < transformInfo.Size(); ++i ) {
		NeoPresume( transformInfo[i] >= BD_BatchLength && transformInfo[i] <= BD_Count );
		if( transformInfo[i] == BD_Count ) {
			outputDesc.SetDimSize( i, 1 );
		} else {
			outputDesc.SetDimSize( i, inputDesc.DimSize( transformInfo[i] ) );
		}
	}

	outputShapeBlobs[0] = inputShapeBlobs[0]->GetCopy();
	outputShapeBlobs[0]->ReinterpretDimensions( outputDesc );
}

} // namespace NeoML
