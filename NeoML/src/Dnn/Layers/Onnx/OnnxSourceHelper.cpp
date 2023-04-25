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

#include <NeoML/Dnn/Layers/Onnx/OnnxSourceHelper.h>

namespace NeoML {

CPtr<CDnnBlob>& COnnxSourceHelper::Blob()
{
	ForceReshape();
	return blob;
}

static const int OnnxSourceHelperVersion = 0;

void COnnxSourceHelper::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxSourceHelperVersion );
	COnnxLayerBase::Serialize( archive );
	SerializeBlob( GetDefaultCpuMathEngine(), archive, blob );
}

void COnnxSourceHelper::CalculateShapes()
{
	CheckLayerArchitecture( GetInputCount() == 0, "OnnxSourceHelper must have no inputs" );
	CheckLayerArchitecture( GetOutputCount() == 1, "OnnxSourceHelper must have 1 output" );
	CheckLayerArchitecture( blob != nullptr, "OnnxSourceHelper with null blob" );

	if( &blob->GetMathEngine() != &MathEngine() ) {
		outputShapeBlobs[0] = CDnnBlob::CreateBlob( MathEngine(), blob->GetDataType(), blob->GetDesc() );
		outputShapeBlobs[0]->CopyFrom( blob );
	} else {
		outputShapeBlobs[0] = blob;
	}
}

} // namespace NeoML
