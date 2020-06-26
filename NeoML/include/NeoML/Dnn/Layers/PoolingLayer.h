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

// The base class for all pooling layers
class NEOML_API CPoolingLayer : public CBaseLayer {
public:
	// The filter size
	int GetFilterHeight() const { return filterHeight; }
	void SetFilterHeight( int filterHeight );
	int GetFilterWidth() const { return filterWidth; }
	void SetFilterWidth( int filterWidth );

	// The filter stride, vertical and horizontal:
	int GetStrideHeight() const { return strideHeight; }
	void SetStrideHeight( int strideHeight );
	int GetStrideWidth() const { return strideWidth; }
	void SetStrideWidth( int strideWidth );

	void Serialize( CArchive& archive ) override;

protected:
	CPoolingLayer( IMathEngine& mathEngine, const char* name );

	int filterHeight;	// the filter height
	int filterWidth;	// the filter width
	int strideHeight;	// the vertical filter stride
	int strideWidth;	// the horizontal filter stride

	void Reshape() override;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CMaxPoolingLayer implements a pooling layer that takes the maximum in the window
class NEOML_API CMaxPoolingLayer : public CPoolingLayer {
	NEOML_DNN_LAYER( CMaxPoolingLayer )
public:
	explicit CMaxPoolingLayer( IMathEngine& mathEngine ) : CPoolingLayer( mathEngine, "CCnnMaxPoolingLayer" ), desc( 0 ) {}

	void Serialize( CArchive& archive ) override;

protected:
	virtual ~CMaxPoolingLayer() { destroyDesc(); }

	void RunOnce() override;
	void BackwardOnce() override;
	void Reshape() override;

private:
	CPtr<CDnnBlob> maxIndices; // contains the maximums' indices (for the backward pass)
	CMaxPoolingDesc* desc;

	void initDesc();
	void destroyDesc();
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CMeanPoolingLayer implements a pooling layer that takes the average over the window
class NEOML_API CMeanPoolingLayer : public CPoolingLayer {
	NEOML_DNN_LAYER( CMeanPoolingLayer )
public:
	explicit CMeanPoolingLayer( IMathEngine& mathEngine ) : CPoolingLayer( mathEngine, "CCnnMeanPoolingLayer" ), desc( 0 ) {}

	void Serialize( CArchive& archive ) override;

protected:
	virtual ~CMeanPoolingLayer() { destroyDesc(); }

	void RunOnce() override;
	void BackwardOnce() override;
	void Reshape() override;

private:
	CMeanPoolingDesc* desc;

	void initDesc();
	void destroyDesc();
};

} // namespace NeoML
