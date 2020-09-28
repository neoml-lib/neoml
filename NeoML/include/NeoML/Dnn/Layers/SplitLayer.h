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

// CBaseSplitLayer is the base layer for splitting a blob into several parts
class NEOML_API CBaseSplitLayer : public CBaseLayer {
public:
	const CArray<int>& GetOutputCounts() const { return outputCounts; }
	void SetOutputCounts(const CArray<int>& _outputCounts);

	// The same as SetOutputCounts for 2, 3, 4 outputs
	void SetOutputCounts2(int count0);
	void SetOutputCounts3(int count0, int count1);
	void SetOutputCounts4(int count0, int count1, int count2);

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void Serialize( CArchive& archive ) override;

protected:
	const TBlobDim dimension;
	CArray<int> outputCounts;

	CBaseSplitLayer( IMathEngine& mathEngine, TBlobDim _dimension, const char* name );
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CSplitChannelsLayer implements a layer that splits a blob by the Channels dimension
class NEOML_API CSplitChannelsLayer : public CBaseSplitLayer {
	NEOML_DNN_LAYER( CSplitChannelsLayer )
public:
	explicit CSplitChannelsLayer( IMathEngine& mathEngine ) : CBaseSplitLayer( mathEngine, BD_Channels, "CCnnSplitChannelsLayer" ) {}

	void Serialize( CArchive& archive ) override;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CSplitDepthLayer implements a layer that splits a blob by the Depth dimension
class NEOML_API CSplitDepthLayer : public CBaseSplitLayer {
	NEOML_DNN_LAYER( CSplitDepthLayer )
public:
	explicit CSplitDepthLayer( IMathEngine& mathEngine ) : CBaseSplitLayer( mathEngine, BD_Depth, "CCnnSplitDepthLayer" ) {}

	void Serialize( CArchive& archive ) override;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CSplitWidthLayer implements a layer that splits a blob by the Width dimension (horizontally)
class NEOML_API CSplitWidthLayer : public CBaseSplitLayer {
	NEOML_DNN_LAYER( CSplitWidthLayer )
public:
	explicit CSplitWidthLayer( IMathEngine& mathEngine ) : CBaseSplitLayer( mathEngine, BD_Width, "CCnnSplitWidthLayer" ) {}

	void Serialize( CArchive& archive ) override;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CSplitHeightLayer implements a layer that splits a blob by the Height dimension (vertically)
class NEOML_API CSplitHeightLayer : public CBaseSplitLayer {
	NEOML_DNN_LAYER( CSplitHeightLayer )
public:
	explicit CSplitHeightLayer( IMathEngine& mathEngine ) : CBaseSplitLayer( mathEngine, BD_Height, "CCnnSplitHeightLayer" ) {}

	void Serialize( CArchive& archive ) override;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CSplitBatchWidthLayer implements a layer that splits a blob by the BatchWidth dimension
class NEOML_API CSplitBatchWidthLayer : public CBaseSplitLayer {
	NEOML_DNN_LAYER( CSplitBatchWidthLayer )
public:
	explicit CSplitBatchWidthLayer( IMathEngine& mathEngine ) : CBaseSplitLayer( mathEngine, BD_BatchWidth, "CCnnSplitBatchWidthLayer" ) {}

	void Serialize( CArchive& archive ) override;
};

} // namespace NeoML
