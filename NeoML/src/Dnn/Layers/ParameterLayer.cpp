/* Copyright © 2024 ABBYY

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
	bool sameBlob = _blob == paramBlobs[0].Ptr();
	paramBlobs[0] = _blob;

	if( !outputDescs.IsEmpty() ) {
		if( _blob != nullptr
			&& ( _blob->GetDataType() != outputDescs[0].GetDataType()
			|| !_blob->GetDesc().HasEqualDimensions( outputDescs[0] ) ) )
		{
			outputDescs[0] = paramBlobs[0]->GetDesc();
			ForceReshape();
		} else {
			sameBlob = false;
		}
	}

	if ( !outputBlobs.IsEmpty() && !sameBlob ) {
		outputBlobs[0] = 0;
	}
}

void CParameterLayer::SetBlobDesc(const CBlobDesc& _desc)
{
	bool isReshapeNeeded = desc.GetDataType() == CT_Invalid
		|| !desc.HasEqualDimensions(_desc)
		|| desc.GetDataType() != _desc.GetDataType();

	desc = _desc;

	if( isReshapeNeeded ) {
		paramBlobs[0] = 0;
		ForceReshape();
		if( !outputBlobs.IsEmpty() ) {
			outputBlobs[0] = 0;
		}
	}
}

void CParameterLayer::Reshape()
{
	CheckOutputs();
	CheckLayerArchitecture(GetInputCount() == 0, "layer must not have inputs");
	CheckLayerArchitecture(GetOutputCount() == 1, "layer has more than 1 output");

	if( paramBlobs[0].Ptr() == nullptr ) {
		paramBlobs[0] = CDnnBlob::CreateBlob(MathEngine(), desc);
		InitializeParamBlob(0, *paramBlobs[0], paramBlobs[0]->GetDataSize());
	}

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

	if(archive.IsStoring()) {
		for(TBlobDim d = TBlobDim(0); d < BD_Count; ++d) {
			archive << desc.DimSize(d);
		}
	} else if(archive.IsLoading()) {
		CBlobDesc desc(CT_Float);
		for(TBlobDim d = TBlobDim(0); d < BD_Count; ++d) {
			int size;
			archive >> size;
			desc.SetDimSize(d, size);
		}
	}
}

CParameterLayer* Parameter( CDnn& dnn, const char* name, CDnnBlob* blob ) {
	CPtr<CParameterLayer> param = new CParameterLayer(dnn.GetMathEngine());
	param->SetName(name);
	param->SetBlob(blob);
	dnn.AddLayer(*param);
	return param;
}

} // namespace NeoML
