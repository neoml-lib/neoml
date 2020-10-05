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

// CProjectionPoolingLayer implements a layer that calculating average along one of the blob axis
class NEOML_API CProjectionPoolingLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CProjectionPoolingLayer )
public:
	explicit CProjectionPoolingLayer( IMathEngine& mathEngine );
	virtual ~CProjectionPoolingLayer();

	// Projection dimension
	// BD_Width by default
	TBlobDim GetDimension() const { return dimension; }
	void SetDimenion( TBlobDim dimension );

	// If true then output size is equal to the input size and pooling result is broadcasted along the projection dimension
	// Otherwise projection dimension of the output is equal to 1 and other dimensions are equal to ones of the input
	// false by default
	bool GetRestoreOriginalImageSize() const { return restoreOriginalImageSize; }
	void SetRestoreOriginalImageSize( bool flag );

protected:
	// CBaseLayer methods
	void Serialize( CArchive& archive ) override;
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	// Projection dimension
	TBlobDim dimension;
	// Does layer preserve input blob's shape?
	bool restoreOriginalImageSize;

	// Temporary blob for pool results
	CPtr<CDnnBlob> projectionResultBlob;

	// Pooling descriptor
	CMeanPoolingDesc* desc;

	void initDesc( const CBlobDesc& inputDesc );
	void destroyDesc();
};

} // namespace NeoML
