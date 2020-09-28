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
#include <NeoML/Dnn/Layers/BaseInPlaceLayer.h>

namespace NeoML {

// CTransformLayer implements a layer that changes the blob dimensions without shifting the data in memory
// For example, you may double the height and halve the width
class NEOML_API CTransformLayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CTransformLayer )
public:
	explicit CTransformLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The operation to be performed over the given dimension
	enum TOperation {
		// Set this dimension so that the total size stays the same
		O_Remainder,
		// Set this dimension to Parameter value
		O_SetSize,
		// Multiply this dimension by Parameter value
		O_Multiply,
		// Divide this dimension by Parameter value
		O_Divide
	};

	// The rule of dimension change
	struct NEOML_API CDimensionRule {
		// The mode of dimension change
		TOperation Operation;
		// The numerical parameter to be used
		int Parameter;

		CDimensionRule();
		CDimensionRule( TOperation op, int param );

		bool operator==( const CDimensionRule& other ) const;

		// Applies the transformation set by the rule
		int Transform( int input ) const;
	};

	// The parameters for transforming the specified dimension
	const CDimensionRule& GetDimensionRule( TBlobDim dim ) const
		{ return rules[dim]; }
	void SetDimensionRule( TBlobDim dim, const CDimensionRule& rule );
	void SetDimensionRule( TBlobDim dim, TOperation op, int param );

protected:
	~CTransformLayer();

	void OnReshaped() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	// Transformation parameters for all dimensions
	CDimensionRule rules[BD_Count];

	// Blob descs
	CBlobDesc inputDesc;
	CBlobDesc outputDesc;
};

} // namespace NeoML

