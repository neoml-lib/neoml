/* Copyright © 2017-2023 ABBYY

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
#include <NeoML/Dnn/DnnBlob.h>
#include <NeoML/Dnn/Rowwise/RowwiseOperation.h>

namespace NeoML {

class CImageResizeLayer;

class NEOML_API CRowwiseImageResize : public IRowwiseOperation {
public:
	// Creates an equivalent of a layer
	explicit CRowwiseImageResize( const CImageResizeLayer& layer );
	// Constructor for serialization
	explicit CRowwiseImageResize( IMathEngine& mathEngine );

	// IRowwiseOperation implementation
	CRowwiseOperationDesc* GetDesc() override;
	void Serialize( CArchive& archive ) override;

private:
	IMathEngine& mathEngine; // math engine used for calculations
	// Parameters of image resize
	TBlobResizePadding padding;
	float defaultValue;
	int deltaLeft;
	int deltaRight;
	int deltaTop;
	int deltaBottom;
};

} // namespace NeoML
