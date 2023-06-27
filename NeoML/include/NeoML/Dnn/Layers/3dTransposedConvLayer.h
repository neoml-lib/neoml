/* Copyright © 2017-2020 ABBYY Production LLC

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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Layers/3dConvLayer.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// C3dTransposedConvLayer implements a transposed convolution layer (aka deconvolution, up-convolution)
class NEOML_API C3dTransposedConvLayer : public CBase3dConvLayer {
	NEOML_DNN_LAYER( C3dTransposedConvLayer )
public:
	explicit C3dTransposedConvLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	~C3dTransposedConvLayer() override { destroyConvDesc(); }

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;
	bool IsFilterTransposed() const override { return true; }
	int BlobsForBackward() const override { return 0; }
	int BlobsForLearn() const override { return TInputBlobs; }

private:
	C3dConvolutionDesc* convDesc;

	void initConvDesc();
	void destroyConvDesc();
	void calcOutputBlobSize( int& outputHeight, int& outputWidth, int& outputDepth ) const;
};

} // namespace NeoML
