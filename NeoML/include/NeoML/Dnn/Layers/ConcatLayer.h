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

// CBaseConcatLayer is a base class for blob concatenation layers
class NEOML_API CBaseConcatLayer : public CBaseLayer {
public:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

	void Serialize( CArchive& archive ) override;

protected:
	const TBlobDim dimension;

	CBaseConcatLayer( IMathEngine& mathEngine, TBlobDim _dimension, const char* name );
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CConcatChannelsLayer implements a layer that concatenates several blobs into one along the Channel dimension
class NEOML_API CConcatChannelsLayer : public CBaseConcatLayer {
	NEOML_DNN_LAYER( CConcatChannelsLayer )
public:
	explicit CConcatChannelsLayer( IMathEngine& mathEngine ) : CBaseConcatLayer( mathEngine, BD_Channels, "CCnnConcatChannelsLayer" ) {}

	void Serialize( CArchive& archive ) override;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CConcatDepthLayer implements a layer that concatenates several blobs into one along the Depth dimension
class NEOML_API CConcatDepthLayer : public CBaseConcatLayer {
	NEOML_DNN_LAYER( CConcatDepthLayer )
public:
	explicit CConcatDepthLayer( IMathEngine& mathEngine ) : CBaseConcatLayer( mathEngine, BD_Depth, "CCnnConcatDepthLayer" ) {}

	void Serialize( CArchive& archive ) override;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CConcatWidthLayer implements a layer that concatenates several blobs into one along the Width dimension (i.e. horizontally)
class NEOML_API CConcatWidthLayer : public CBaseConcatLayer {
	NEOML_DNN_LAYER( CConcatWidthLayer )
public:
	explicit CConcatWidthLayer( IMathEngine& mathEngine ) : CBaseConcatLayer( mathEngine, BD_Width, "CCnnConcatWidthLayer" ) {}

	void Serialize( CArchive& archive ) override;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CConcatHeightLayer implements a layer that concatenates several blobs into one along the Height dimension (i.e. vertically)
class NEOML_API CConcatHeightLayer : public CBaseConcatLayer {
	NEOML_DNN_LAYER( CConcatHeightLayer )
public:
	explicit CConcatHeightLayer( IMathEngine& mathEngine ) : CBaseConcatLayer( mathEngine, BD_Height, "CCnnConcatHeightLayer" ) {}

	void Serialize( CArchive& archive ) override;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CConcatBatchLayer implements a layer that concatenates several blobs into one 
// along the BatchWidth dimension (simply stores them one after another)
class NEOML_API CConcatBatchWidthLayer : public CBaseConcatLayer {
	NEOML_DNN_LAYER( CConcatBatchWidthLayer )
public:
	explicit CConcatBatchWidthLayer( IMathEngine& mathEngine ) : CBaseConcatLayer( mathEngine, BD_BatchWidth, "CCnnConcatBatchWidthLayer" ) {}

	void Serialize( CArchive& archive ) override;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CConcatObjectLayer implements a layer that concatenates several blobs into one by objects
// An object may be of any configuration; the raw data is concatenated
// The result will be an array of 1*1*1*Channels dimensions
class NEOML_API CConcatObjectLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CConcatObjectLayer )
public:
	explicit CConcatObjectLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CCnnConcatObjectLayer", false ) {}

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

} // namespace NeoML
