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
#include <NeoML/Dnn/Layers/BaseInPlaceLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// Takes single integer blob of any size as an input.
// The only output contains integer blob of the same size where
//    output[i] = input[i] == 0 ? 1 : 0
class NEOML_API CNotLayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CNotLayer )
public:
	explicit CNotLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	void OnReshaped() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

NEOML_API CLayerWrapper<CNotLayer> Not();

// --------------------------------------------------------------------------------------------------------------------

// Takes 2 blobs of the same size and data type
// The only outputs contains integer blob of the same size where
//    outputs[i] = input0[i] < input1[i] ? 1 : 0
class NEOML_API CLessLayer : public CEltwiseBaseLayer {
	NEOML_DNN_LAYER( CLessLayer )
public:
	explicit CLessLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

NEOML_API CLayerWrapper<CLessLayer> Less();

// --------------------------------------------------------------------------------------------------------------------

// Takes 2 blobs of the same size and data type
// The only outputs contains integer blob of the same size where
//    outputs[i] = input0[i] == input1[i] ? 1 : 0
class NEOML_API CEqualLayer : public CEltwiseBaseLayer {
	NEOML_DNN_LAYER( CEqualLayer )
public:
	explicit CEqualLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

NEOML_API CLayerWrapper<CEqualLayer> Equal();

// --------------------------------------------------------------------------------------------------------------------

// Takes 3 blobs of the same size
// The first input must be of integer data
// The second and third input may be of any type (type must be the same)
// The only outputs contains integer blob of the same size where
//    outputs[i] = input0[i] != 0 ? input1[i] : input2[i]
class NEOML_API CWhereLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CWhereLayer )
public:
	explicit CWhereLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

NEOML_API CLayerWrapper<CWhereLayer> Where();

} // namespace NeoML
