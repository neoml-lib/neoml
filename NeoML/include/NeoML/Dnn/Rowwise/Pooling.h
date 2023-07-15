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

class CMaxPoolingLayer;
class CMeanPoolingLayer;

class NEOML_API CRowwise2DPooling : public IRowwiseOperation {
public:
	// Creates an equivalent of a layer
	explicit CRowwise2DPooling( const CMaxPoolingLayer& layer );
	explicit CRowwise2DPooling( const CMeanPoolingLayer& layer );
	// Constructor for serialization
	explicit CRowwise2DPooling( IMathEngine& mathEngine );

	// IRowwiseOperation implementation
	CRowwiseOperationDesc* GetDesc() override
	{ return mathEngine.InitRowwise2DPooling( isMax, filterHeight, filterWidth, strideHeight, strideWidth ); }

	void Serialize( CArchive& archive ) override;

public:
	IMathEngine& mathEngine; // math engine used for calculations
	// Parameters of pooling
	bool isMax;
	int filterHeight;
	int filterWidth;
	int strideHeight;
	int strideWidth;
};

} // namespace NeoML
