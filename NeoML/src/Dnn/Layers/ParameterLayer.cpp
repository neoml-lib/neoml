/* Copyright © 2024 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/ParameterLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

void CParameterLayer::AllocateOutputBlobs()
{
	outputBlobs[0] = paramBlobs[0];
}

void CParameterLayer::SetBlob(CDnnBlob* _blob)
{
	if (_blob == paramBlobs[0].Ptr()) {
		return;
	}

	paramBlobs[0] = _blob;

	if (!outputDescs.IsEmpty()) {
		if (paramBlobs[0]->GetDataType() != outputDescs[0].GetDataType()
			|| !paramBlobs[0]->GetDesc().HasEqualDimensions(outputDescs[0]))
		{
			outputDescs[0] = paramBlobs[0]->GetDesc();
			ForceReshape();
		}
	}

	if (!outputBlobs.IsEmpty()) {
		outputBlobs[0] = 0;
	}
}

void CParameterLayer::Reshape()
{
	CheckOutputs();
	CheckLayerArchitecture(GetInputCount() == 0, "layer must not have inputs");
	CheckLayerArchitecture(GetOutputCount() == 1, "layer has more than 1 output");
	CheckLayerArchitecture(paramBlobs[0].Ptr() != nullptr, "layer has null param blob");
	outputDescs[0] = paramBlobs[0]->GetDesc();
}

void CParameterLayer::RunOnce()
{
	// Is done while AllocateOutputBlobs
}

void CParameterLayer::LearnOnce()
{
	// Layer's derivative is one => equal to outputDiff
	paramDiffBlobs[0]->Add( outputDiffBlobs[0] );
}

void CParameterLayer::BackwardOnce()
{
	// Skip for this layer
}

void CParameterLayer::Serialize(CArchive& archive)
{
	archive.SerializeVersion( 0 );
	CBaseLayer::Serialize(archive);
}

CParameterLayer* Parameter( CDnn& dnn, const char* name, CDnnBlob* blob ) {
	CPtr<CParameterLayer> param = new CParameterLayer(dnn.GetMathEngine());
	param->SetName(name);
	param->SetBlob(blob);
	dnn.AddLayer(*param);
	return param;
}

} // namespace NeoML
