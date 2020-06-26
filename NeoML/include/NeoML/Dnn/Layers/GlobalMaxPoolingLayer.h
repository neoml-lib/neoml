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

// CGlobalMaxPoolingLayer implements a layer that selects the given number of largest elements 
// over the whole image (separately in each channel)
// If the input blob contains several multi-channel images of Height*Width size,
// the input blob will have the same dimensions except that Height = 1, Width = the number of largest elements
class NEOML_API CGlobalMaxPoolingLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CGlobalMaxPoolingLayer )
public:
	explicit CGlobalMaxPoolingLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	int GetMaxCount() const { return maxCount; }
	void SetMaxCount(int _enumSize);

protected:
	virtual ~CGlobalMaxPoolingLayer() { destroyDesc(); }

	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	CGlobalMaxPoolingDesc* desc;
	// The number of largest elements to be found
	int maxCount;
	// the blob with the largest elements' indices
	CPtr<CDnnBlob> indexBlob;

	void initDesc();
	void destroyDesc();
};

} // namespace NeoML
