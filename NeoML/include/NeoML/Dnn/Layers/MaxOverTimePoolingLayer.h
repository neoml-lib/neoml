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

// The layer that performs max-over-time pooling.
// A blob of L x B x List x H x W x D x C dimensions is converted 
// into a blob of New_L x B x List x H x W x D x C dimensions 
// by finding the maximum element in the window over the first dimension.
class NEOML_API CMaxOverTimePoolingLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CMaxOverTimePoolingLayer )
public:
	explicit CMaxOverTimePoolingLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The window length and stride. If the length is <=0, global max pooling is performed (over the whole dimension)
	int GetFilterLength() const { return filterLength; }
	void SetFilterLength(int length);
	int GetStrideLength() const { return strideLength; }
	void SetStrideLength(int length);

protected:
	virtual ~CMaxOverTimePoolingLayer() { destroyDescs(); }

	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	CMaxOverTimePoolingDesc* desc;
	CGlobalMaxOverTimePoolingDesc* globalDesc;
	int filterLength;	// the filter length
	int strideLength;	// the filter stride
	CPtr<CDnnBlob> maxIndices; // the indices of the maximum elements (for the backward pass)

	void initDescs();
	void destroyDescs();
};

} // namespace NeoML
