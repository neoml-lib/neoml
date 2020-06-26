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

class NEOML_API C3dPoolingLayer : public CBaseLayer {
public:
	// the size of the pooling window
	int GetFilterHeight() const { return filterHeight; }
	void SetFilterHeight(int filterHeight);
	int GetFilterWidth() const { return filterWidth; }
	void SetFilterWidth(int filterWidth);
	int GetFilterDepth() const { return filterDepth; }
	void SetFilterDepth(int filterDepth);

	// the filter stride, vertical and horizontal
	int GetStrideHeight() const { return strideHeight; }
	void SetStrideHeight(int strideHeight);
	int GetStrideWidth() const { return strideWidth; }
	void SetStrideWidth(int strideWidth);
	int GetStrideDepth() const { return strideDepth; }
	void SetStrideDepth(int strideDepth);

	void Serialize( CArchive& archive ) override;

protected:
	C3dPoolingLayer( IMathEngine& mathEngine, const char* name );

	void Reshape() override;

	int filterHeight;	// window height
	int filterWidth;	// window width
	int filterDepth;	// window depth
	int strideHeight;	// filter stride along the height dimension
	int strideWidth;	// filter stride along the width dimension
	int strideDepth;	// filter stride along the depth dimension
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// C3dMaxPoolingLayer implements a pooling layer that takes the maximum value in the window
class NEOML_API C3dMaxPoolingLayer : public C3dPoolingLayer {
	NEOML_DNN_LAYER( C3dMaxPoolingLayer )
public:
	explicit C3dMaxPoolingLayer( IMathEngine& mathEngine ) : C3dPoolingLayer( mathEngine, "CCnn3dMaxPoolingLayer" ), desc( 0 ) {}

	void Serialize( CArchive& archive ) override;

protected:
	virtual ~C3dMaxPoolingLayer() { destroyDesc(); }

	void RunOnce() override;
	void BackwardOnce() override;
	void Reshape() override;

private:
	CPtr<CDnnBlob> indexBlob; // the indices of maximum elements, used for backpropagation
	C3dMaxPoolingDesc* desc;

	void initDesc();
	void destroyDesc();
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// C3dMeanPoolingLayer implements a pooling layer that takes a mean value in the window
class NEOML_API C3dMeanPoolingLayer : public C3dPoolingLayer {
	NEOML_DNN_LAYER( C3dMeanPoolingLayer )
public:
	explicit C3dMeanPoolingLayer( IMathEngine& mathEngine ) : C3dPoolingLayer( mathEngine, "CCnn3dMeanPoolingLayer" ), desc( 0 ) {}
	
	void Serialize( CArchive& archive ) override;

protected:
	virtual ~C3dMeanPoolingLayer() { destroyDesc(); }

	void RunOnce() override;
	void BackwardOnce() override;
	void Reshape() override;

private:
	C3dMeanPoolingDesc* desc;

	void initDesc();
	void destroyDesc();
};

} // namespace NeoML
