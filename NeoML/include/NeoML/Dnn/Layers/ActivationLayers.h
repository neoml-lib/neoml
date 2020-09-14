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
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// Supported activation functions
enum TActivationFunction {
	AF_Linear = 0,
	AF_ELU,
	AF_ReLU,
	AF_LeakyReLU,
	AF_Abs,
	AF_Sigmoid,
	AF_Tanh,
	AF_HardTanh,
	AF_HardSigmoid,
	AF_Power,
	AF_HSwish,
	AF_GELU,

	AF_Count
};

// Creates an activation layer using the specified activation function
CPtr<CBaseLayer> NEOML_API CreateActivationLayer( IMathEngine& mathEngine, TActivationFunction type );

//------------------------------------------------------------------------------------------------------------

// The layer that uses a linear activation function a*x + b
class NEOML_API CLinearLayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CLinearLayer )
public:
	explicit CLinearLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	float GetMultiplier() const { return multiplier; }
	void SetMultiplier(float _multiplier) { multiplier = _multiplier; }
	float GetFreeTerm() const { return freeTerm; }
	void SetFreeTerm(float _freeTerm) { freeTerm = _freeTerm; }

protected:
	float multiplier;
	float freeTerm;

	void RunOnce() override;
	void BackwardOnce() override;
};

//------------------------------------------------------------------------------------------------------------

// The layer that uses ELU activation function:
// f(x) = x if x >= 0
// f(x) = alpha * (exp(x) - 1) if x < 0
class NEOML_API CELULayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CELULayer )
public:
	explicit CELULayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	float GetAlpha() const;
	void SetAlpha( float newAlpha );

protected:
	void RunOnce() override;
	void BackwardOnce() override;
};

//------------------------------------------------------------------------------------------------------------

// The layer that uses ReLU activation function: f(x) = max(0, x)
class NEOML_API CReLULayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CReLULayer )
public:
	explicit CReLULayer( IMathEngine& mathEngine ) : CBaseInPlaceLayer( mathEngine, "CCnnReLULayer" ), upperThreshold( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) )
		{ SetUpperThreshold( 0.0 ); }

	void Serialize( CArchive& archive ) override;

	// The upper cutoff for the function value. If you set it to a value > 0, 
	// the function will be ReLU(x) = Upper_Threshold for x > Upper_Threshold
	// The default value is 0: no cutoff
	float GetUpperThreshold() const { return upperThreshold->GetData().GetValue(); }
	void SetUpperThreshold(float threshold) { upperThreshold->GetData().SetValue(threshold); }

protected:
	void RunOnce() override;
	void BackwardOnce() override;

private:
	CPtr<CDnnBlob> upperThreshold;
};

//------------------------------------------------------------------------------------------------------------

// The layer that uses "leaky ReLU" activation function:
// f(x) = x if x > 0
// f(x) = alpha * x if x < 0
class NEOML_API CLeakyReLULayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CLeakyReLULayer )
public:
	explicit CLeakyReLULayer( IMathEngine& mathEngine );

	float GetAlpha() const;
	void SetAlpha( float newAlpha );

	void Serialize( CArchive& archive ) override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
};

//------------------------------------------------------------------------------------------------------------

// The layer that uses the activation function f(x) = x * ReLU6(x + 3) / 6 
class NEOML_API CHSwishLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CHSwishLayer )
public:
	explicit CHSwishLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CHSwishLayer", false ) {}

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

//------------------------------------------------------------------------------------------------------------

// The layer that uses abs(x) activation function
class NEOML_API CAbsLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CAbsLayer )
public:
	explicit CAbsLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CCnnAbsLayer", false ) {}

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

//------------------------------------------------------------------------------------------------------------

// The layer that uses a sigmoid activation function 1 / (1 + exp(-x))
class NEOML_API CSigmoidLayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CSigmoidLayer )
public:
	explicit CSigmoidLayer( IMathEngine& mathEngine ) : CBaseInPlaceLayer( mathEngine, "CCnnSigmoidLayer" ) {}

	void Serialize( CArchive& archive ) override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
};

//------------------------------------------------------------------------------------------------------------

// The layer that uses tanh(x) activation function
class NEOML_API CTanhLayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CTanhLayer )
public:
	explicit CTanhLayer( IMathEngine& mathEngine ) : CBaseInPlaceLayer( mathEngine, "CCnnTanhLayer" ) {}

	void Serialize( CArchive& archive ) override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
};

//------------------------------------------------------------------------------------------------------------

// The layer that uses HardTanh activation function:
// HardTanh(x) = { -1 : x <= -1; x : -1 < x < 1; 1 : x >= 1 }
class NEOML_API CHardTanhLayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CHardTanhLayer )
public:
	explicit CHardTanhLayer( IMathEngine& mathEngine ) : CBaseInPlaceLayer( mathEngine, "CCnnHardTanhLayer" ) {}

	void Serialize( CArchive& archive ) override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
};

//------------------------------------------------------------------------------------------------------------

// The layer that uses HardSigmoid activation function:
// HardSigmoid(x) = { 0 : x <= 0; x : 0 < x < 1; 1 : x >= 1 }
class NEOML_API CHardSigmoidLayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CHardSigmoidLayer )
public:
	explicit CHardSigmoidLayer( IMathEngine& mathEngine );

	float GetSlope() const { return paramBlobs[0]->GetData().GetValue(); }
	void SetSlope( float slope ) { paramBlobs[0]->GetData().SetValue( slope ); }
	float GetBias() const { return paramBlobs[1]->GetData().GetValue(); }
	void SetBias( float bias ) { paramBlobs[1]->GetData().SetValue( bias ); }

	void Serialize( CArchive& archive ) override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;

private:
	void setDefaultParamBlobs( IMathEngine& mathEngine );
};

//------------------------------------------------------------------------------------------------------------

// The layer that raises each element to the given power
class NEOML_API CPowerLayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CPowerLayer )
public:
	explicit CPowerLayer( IMathEngine& mathEngine ) : CBaseInPlaceLayer( mathEngine, "CCnnPowerLayer" ), exponent(0) {}

	void Serialize( CArchive& archive ) override;

	void SetExponent( float newExponent ) { exponent = newExponent; }
	float GetExponent() const { return exponent; }

protected:
	void RunOnce() override;
	void BackwardOnce() override;

private:
	float exponent; // the power to which the elements will be raised
};

} // namespace NeoML
