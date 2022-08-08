/* Copyright © 2017-2022 ABBYY Production LLC

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

// Layer that calculates cumulative sum over the blob dimension
//    CumSum(x)[i] = x[0] + x[1] + ... + x[i]
class NEOML_API CCumSumLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CCumSumLayer )
public:
	CCumSumLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The dimension along which sum is cumulated
	void SetDimension( TBlobDim newDim ) { dim = newDim; }
	TBlobDim GetDimension() const { return dim; }

	// Sets the direction to cumulate sum along the dimension (direct or reverse)
	void SetReverse( bool newReverse ) { reverse = newReverse; }
	bool IsReverse() const { return reverse; }

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	TBlobDim dim;
	bool reverse;
};

NEOML_API CLayerWrapper<CCumSumLayer> CumSum( TBlobDim dim, bool reverse );

} // namespace NeoML
