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

#include <NeoML/Dnn/Layers/Onnx/ShapeToBlobLayer.h>
#include <NeoML/Dnn/Layers/Onnx/BaseReshaper.h>

namespace NeoML {

static const int ShapeToBlobLayerVersion = 0;

void CShapeToBlobLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ShapeToBlobLayerVersion );
	CBaseLayer::Serialize( archive );
}

void CShapeToBlobLayer::Reshape()
{
	CheckArchitecture( GetInputCount() == 1, GetPath(), "Layer must have 1 input" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "Layer must have 1 output" );
	const CBaseReshaper* shapeProvider = dynamic_cast<const CBaseReshaper*>( GetInputLayer( 0 ) );
	CheckArchitecture( shapeProvider != nullptr, GetPath(), "Input must contain shape" );
	CheckArchitecture( shapeProvider->GetOutputShapeBlobs().IsValidIndex( GetInputOutputNumber( 1 ) ), GetPath(),
		"Wrong input number" );
	outputDescs[0] = shapeProvider->GetOutputShapeBlobs()[GetInputOutputNumber( 1 )]->GetDesc();
}

void CShapeToBlobLayer::RunOnce()
{
	const CBaseReshaper* shapeProvider = dynamic_cast<const CBaseReshaper*>( GetInputLayer( 0 ) );
	NeoPresume( shapeProvider != nullptr );
	CPtr<CDnnBlob> inputShapeBlob = shapeProvider->GetOutputShapeBlobs()[GetInputOutputNumber( 1 )];
	NeoPresume( outputBlobs[0]->HasEqualDimensions( inputShapeBlob.Ptr() ) );
	outputBlobs[0]->CopyFrom( inputShapeBlob );
}

} // namespace NeoML
