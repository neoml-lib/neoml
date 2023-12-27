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

 #include <NeoML/Dnn/Layers/TrainableTransformLayer.h>
 #include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

void CTrainableTransformLayer::AllocateOutputBlobs()
{
	outputBlobs[0] = paramBlobs[0];
}

void CTrainableTransformLayer::SetBlob(CDnnBlob* _blob) {
	if (_blob == paramBlobs[0].Ptr()) {
		return;
	}

	paramBlobs[0]->CopyFrom(_blob);

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

void CTrainableTransformLayer::Reshape() {
	outputDescs[0] = paramBlobs[0]->GetDesc();
}

void CTrainableTransformLayer::RunOnce() {
	// Output is equal to the layer's params
	outputBlobs[0]->CopyFrom( paramBlobs[0] );
}

void CTrainableTransformLayer::LearnOnce() {
	// Layer's derivative is one => equal to outputDiff
	paramDiffBlobs[0]->Add( outputDiffBlobs[0] );
}

void CTrainableTransformLayer::BackwardOnce()
{
	// Skip for this layer
}

 } // namespace NeoML