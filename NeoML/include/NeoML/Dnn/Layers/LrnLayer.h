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
//
// LRN(x)[obj][ch] = x[obj][ch] * / ((bias + alpha * sqrSum[obj][ch] / windowSize) ^ beta)
//
// where:
//
// - `obj` is index of object `[0; BlobSize / Channels)`
// - `ch` is index of channel `[0; Channels)` 
// - `windowSize`, `bias`, `alpha`, `beta` are settings
// - `sqrSum` is calculated using the following formula:
//
// sqrSum(x)[obj][ch] = sum(x[obj][i] * x[obj][i] for each i in [ch_min, ch_max])
// ch_min = max(0, ch - floor((windowSize - 1)/2))
// ch_max = min(C - 1, ch + ceil((windowSize - 1)/2))
//
class NEOML_API CLrnLayer: public CBaseLayer {
	NEOML_DNN_LAYER( CLrnLayer )
public:
	explicit CLrnLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Window size, along which sum of squares is calculated
	void SetWindowSize( int value );
	int GetWindowSize() const { return windowSize; }

	// Bias, added to the scaled sum of squares
	void SetBias( float value );
	float GetBias() const { return bias; }

	// Scale of sum of squares
	void SetAlpha( float value );
	float GetAlpha() const { return alpha; }

	// Exponent used in formula
	void SetBeta( float value );
	float GetBeta() const { return beta; }

protected:
	~CLrnLayer() override { destroyDesc(); }

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TInputBlobs | TOutputBlobs; }

private:
	CLrnDesc* desc; // the LRN descriptor
	int windowSize; // size of window, along which sqr_sum is calculated
	float bias; // bias, added to the scaled sum
	float alpha; // scale
	float beta; // exponent

	// temporary blobs
	// used for computation speed-up during backward propagation
	CPtr<CDnnBlob> invertedSum; // 1 / (bias + alpha * sqr_sum / windowSize )
	CPtr<CDnnBlob> invertedSumBeta; // 1 / ((bias + alpha * sqr_sum / windowSize ) ^ beta)

	void initDesc();
	void destroyDesc();
};

NEOML_API CLayerWrapper<CLrnLayer> Lrn( int windowSize, float bias = 1.f, float alpha = 1e-4f, float beta = 0.75f );

} // namespace NeoML
