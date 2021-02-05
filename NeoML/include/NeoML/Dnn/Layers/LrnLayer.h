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
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// Local Response Normalization layer

// LRN(X) = X * (bias + alpha * channelwise-mean-pool(X^2))^(-beta)
// where channelwise-mean-pool is a 1-dimensional mean pooling with filter size equal to
// windowSize and (windowSize-1) / 2 front and back paddings.
class NEOML_API CLrnLayer: public CBaseLayer {
	NEOML_DNN_LAYER( CLrnLayer )
public:
	explicit CLrnLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	void SetWindowSize( int value );
	int GetWindowSize() const { return windowSize; }

	void SetBias( float value );
	float GetBias() const { return bias; }

	void SetAlpha( float value );
	float GetAlpha() const { return alpha; }

	void SetBeta( float value );
	float GetBeta() const { return beta; }

protected:
	virtual ~CLrnLayer() { destroyDesc(); }

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	CLrnDesc* desc; // the LRN descriptor
	int windowSize;
	float bias;
	float alpha;
	float beta;

	CPtr<CDnnBlob> invertedSum;
	CPtr<CDnnBlob> invertedSumBeta;

	void initDesc();
	void destroyDesc();
};

NEOML_API CLayerWrapper<CLrnLayer> Lrn( int windowSize, float bias = 1.f, float alpha = 1e-4f, float beta = 0.75f );

} // namespace NeoML
