/* Copyright © 2017-2022 ABBYY Production LLC

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

static const int OnnxSourceHelperVersion = 0;

void COnnxSourceHelper::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxSourceHelperVersion );
	COnnxLayerBase::Serialize( archive );
	SerializeBlob( GetDefaultCpuMathEngine(), archive, blob );
}

void COnnxSourceHelper::CalculateShapes()
{
	CheckArchitecture( GetInputCount() == 0, GetPath(), "OnnxSourceHelper must have no inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "OnnxSourceHelper must have 1 output" );
	CheckArchitecture( blob != nullptr, GetPath(), "OnnxSourceHelper with null blob" );

	if( &blob->GetMathEngine() != &GetSingleThreadCpuMathEngine() ) {
		outputShapeBlobs[0] = CDnnBlob::CreateBlob( GetSingleThreadCpuMathEngine(),
			blob->GetDataType(), blob->GetDesc() );
		outputShapeBlobs[0]->CopyFrom( blob );
	} else {
		outputShapeBlobs[0] = blob->GetCopy();
	}
}

} // namespace NeoML