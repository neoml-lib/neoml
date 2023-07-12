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
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/BaseInPlaceLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

class CActivationDesc;

// The layer that uses a linear activation function a*x + b
class NEOML_API CLinearLayer : public CBaseInPlaceLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CLinearLayer )
public:
	using CParam = CLinearActivationParam;
	static constexpr float DefaultMultiplier = CParam::DefaultMultiplier;
	static constexpr float DefaultFreeTerm = CParam::DefaultFreeTerm;

	explicit CLinearLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	float GetMultiplier() const { return multiplier; }
	void SetMultiplier( float _multiplier ) { multiplier = _multiplier; }
	float GetFreeTerm() const { return freeTerm; }
	void SetFreeTerm( float _freeTerm ) { freeTerm = _freeTerm; }

	void ApplyParam( CParam param ) { SetMultiplier( param.Multiplier ); SetFreeTerm( param.FreeTerm ); }
	CActivationDesc GetDesc() const override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return 0; }

private:
	float multiplier = DefaultMultiplier;
	float freeTerm = DefaultFreeTerm;
};

NEOML_API CLayerWrapper<CLinearLayer> Linear( float multiplier, float freeTerm );

//------------------------------------------------------------------------------------------------------------

// The layer that uses ELU activation function:
// f(x) = x if x >= 0
// f(x) = alpha * (exp(x) - 1) if x < 0
class NEOML_API CELULayer : public CBaseInPlaceLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CELULayer )
public:
	using CParam = CELUActivationParam;
	static constexpr float DefaultAlpha = CParam::DefaultAlpha;

	explicit CELULayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	float GetAlpha() const;
	void SetAlpha( float newAlpha );

	void ApplyParam( CParam param ) { SetAlpha( param.Alpha ); }
	CActivationDesc GetDesc() const override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TOutputBlobs; }
};

NEOML_API CLayerWrapper<CELULayer> Elu( float alpha = CELULayer::DefaultAlpha );

//------------------------------------------------------------------------------------------------------------

// The layer that uses ReLU activation function: f(x) = max(0, x)
class NEOML_API CReLULayer : public CBaseInPlaceLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CReLULayer )
public:
	using CParam = CReLUActivationParam;
	static constexpr float DefaultUpperThreshold = CParam::DefaultUpperThreshold;

	explicit CReLULayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The upper cutoff for the function value. If you set it to a value > 0, 
	// the function will be ReLU(x) = Upper_Threshold for x > Upper_Threshold
	// The default value is 0: no cutoff
	float GetUpperThreshold() const { return upperThreshold->GetData().GetValue(); }
	void SetUpperThreshold( float threshold ) { upperThreshold->GetData().SetValue( threshold ); }

	void ApplyParam( CParam param ) { SetUpperThreshold( param.UpperThreshold ); }
	CActivationDesc GetDesc() const override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TOutputBlobs; }

private:
	CPtr<CDnnBlob> upperThreshold;
};

NEOML_API CLayerWrapper<CReLULayer> Relu( float threshold = CReLULayer::DefaultUpperThreshold );

//------------------------------------------------------------------------------------------------------------

// The layer that uses "leaky ReLU" activation function:
// f(x) = x if x > 0
// f(x) = alpha * x if x < 0
class NEOML_API CLeakyReLULayer : public CBaseInPlaceLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CLeakyReLULayer )
public:
	using CParam = CLeakyReLUActivationParam;
	static constexpr float DefaultAlpha = CParam::DefaultAlpha;

	explicit CLeakyReLULayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	float GetAlpha() const;
	void SetAlpha( float newAlpha );

	void ApplyParam( CParam param ) { SetAlpha( param.Alpha ); }
	CActivationDesc GetDesc() const override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TOutputBlobs; }
};

NEOML_API CLayerWrapper<CLeakyReLULayer> LeakyRelu( float alpha = CLeakyReLULayer::DefaultAlpha );

//------------------------------------------------------------------------------------------------------------

// The layer that uses the activation function f(x) = x * ReLU6(x + 3) / 6 
class NEOML_API CHSwishLayer : public CBaseLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CHSwishLayer )
public:
	explicit CHSwishLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CHSwishLayer", false ) {}

	void Serialize( CArchive& archive ) override;

	CActivationDesc GetDesc() const override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TInputBlobs; }
};

NEOML_API CLayerWrapper<CHSwishLayer> HSwish();

//------------------------------------------------------------------------------------------------------------

// The layer that uses abs(x) activation function
class NEOML_API CAbsLayer : public CBaseLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CAbsLayer )
public:
	explicit CAbsLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CCnnAbsLayer", false ) {}

	void Serialize( CArchive& archive ) override;

	CActivationDesc GetDesc() const override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TInputBlobs; }
};

NEOML_API CLayerWrapper<CAbsLayer> Abs();

//------------------------------------------------------------------------------------------------------------

// The layer that uses a sigmoid activation function 1 / (1 + exp(-x))
class NEOML_API CSigmoidLayer : public CBaseInPlaceLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CSigmoidLayer )
public:
	explicit CSigmoidLayer( IMathEngine& mathEngine ) : CBaseInPlaceLayer( mathEngine, "CCnnSigmoidLayer" ) {}

	void Serialize( CArchive& archive ) override;

	CActivationDesc GetDesc() const override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TOutputBlobs; }
};

NEOML_API CLayerWrapper<CSigmoidLayer> Sigmoid();

//------------------------------------------------------------------------------------------------------------

// The layer that uses tanh(x) activation function
class NEOML_API CTanhLayer : public CBaseInPlaceLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CTanhLayer )
public:
	explicit CTanhLayer( IMathEngine& mathEngine ) : CBaseInPlaceLayer( mathEngine, "CCnnTanhLayer" ) {}

	void Serialize( CArchive& archive ) override;

	CActivationDesc GetDesc() const override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TOutputBlobs; }
};

NEOML_API CLayerWrapper<CTanhLayer> Tanh();

//------------------------------------------------------------------------------------------------------------

// The layer that uses HardTanh activation function:
// HardTanh(x) = { -1 : x <= -1; x : -1 < x < 1; 1 : x >= 1 }
class NEOML_API CHardTanhLayer : public CBaseInPlaceLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CHardTanhLayer )
public:
	explicit CHardTanhLayer( IMathEngine& mathEngine ) : CBaseInPlaceLayer( mathEngine, "CCnnHardTanhLayer" ) {}

	void Serialize( CArchive& archive ) override;

	CActivationDesc GetDesc() const override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TOutputBlobs; }
};

NEOML_API CLayerWrapper<CHardTanhLayer> HardTanh();

//------------------------------------------------------------------------------------------------------------

// The layer that uses HardSigmoid activation function:
// HardSigmoid(x) = { 0 : x <= 0; x : 0 < x < 1; 1 : x >= 1 }
class NEOML_API CHardSigmoidLayer : public CBaseInPlaceLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CHardSigmoidLayer )
public:
	using CParam = CHardSigmoidActivationParam;
	static constexpr float DefaultSlope = CParam::DefaultSlope;
	static constexpr float DefaultBias = CParam::DefaultBias;

	explicit CHardSigmoidLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	float GetSlope() const { return paramBlobs[0]->GetData().GetValue(); }
	void SetSlope( float slope ) { paramBlobs[0]->GetData().SetValue( slope ); }
	float GetBias() const { return paramBlobs[1]->GetData().GetValue(); }
	void SetBias( float bias ) { paramBlobs[1]->GetData().SetValue( bias ); }

	void ApplyParam( CParam param ) { SetSlope( param.Slope ); SetBias( param.Bias ); }
	CActivationDesc GetDesc() const override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TOutputBlobs; }

private:
	void setDefaultParamBlobs( IMathEngine& mathEngine );
};

NEOML_API CLayerWrapper<CHardSigmoidLayer> HardSigmoid( float slope, float bias );

//------------------------------------------------------------------------------------------------------------

// The layer that raises each element to the given power
class NEOML_API CPowerLayer : public CBaseInPlaceLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CPowerLayer )
public:
	using CParam = CPowerActivationParam;
	static constexpr float DefaultExponent = CParam::DefaultExponent;
	explicit CPowerLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	void SetExponent( float newExponent ) { exponent = newExponent; }
	float GetExponent() const { return exponent; }

	void ApplyParam( CParam param ) { SetExponent( param.Exponent ); }
	CActivationDesc GetDesc() const override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TOutputBlobs; }

private:
	float exponent = DefaultExponent; // the power to which the elements will be raised
};

NEOML_API CLayerWrapper<CPowerLayer> Power( float exponent );

//------------------------------------------------------------------------------------------------------------

// The layer that calculates exponent of each element of the input
class NEOML_API CExpLayer : public CBaseInPlaceLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CExpLayer )
public:
	explicit CExpLayer( IMathEngine& mathEngine ) : CBaseInPlaceLayer( mathEngine, "CExpLayer" ) {}

	void Serialize( CArchive& archive ) override;

	CActivationDesc GetDesc() const override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TOutputBlobs; }
};

NEOML_API CLayerWrapper<CExpLayer> Exp();

//------------------------------------------------------------------------------------------------------------

// The layer that calculates logarithm of each element of the input
class NEOML_API CLogLayer : public CBaseInPlaceLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CLogLayer )
public:
	explicit CLogLayer( IMathEngine& mathEngine ) : CBaseInPlaceLayer( mathEngine, "CLogLayer" ) {}

	void Serialize( CArchive& archive ) override;

	CActivationDesc GetDesc() const override;

protected:
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TOutputBlobs; }
};

NEOML_API CLayerWrapper<CLogLayer> Log();

//------------------------------------------------------------------------------------------------------------

// The layer that calculates error function of each element of the input
class NEOML_API CErfLayer : public CBaseLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CErfLayer )
public:
	explicit CErfLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CErfLayer", false ) {}

	void Serialize( CArchive& archive ) override;

	CActivationDesc GetDesc() const override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TInputBlobs; }
};

NEOML_API CLayerWrapper<CErfLayer> Erf();

//------------------------------------------------------------------------------------------------------------

// Activation layer with formula x * Ф(x),
// where Ф(x) - cumulative distribution function of the standard normal distribution N(0, 1)
class NEOML_API CGELULayer : public CBaseLayer, public IActivationLayer {
	NEOML_DNN_LAYER( CGELULayer )
public:
	using CParam = CGELUActivationParam;
	using TCalculationMode = CParam::TCalculationMode;
	static const TCalculationMode DefaultCalculationMode = CParam::DefaultCalculationMode;
	// Backward compatibility
	static_assert( static_cast<int>( TCalculationMode::Count ) == 2, "CGELULayer::TCalculationMode::Count != 2" );
	static const TCalculationMode CM_Precise = CParam::TCalculationMode::Precise;
	static const TCalculationMode CM_SigmoidApproximate = CParam::TCalculationMode::SigmoidApproximate;

	explicit CGELULayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Changes GELU calculation mode
	void SetCalculationMode( TCalculationMode );
	// Returns current calculation mode
	TCalculationMode GetCalculationMode() const { return mode; }

	void ApplyParam( CParam param ) { SetCalculationMode( param.Mode ); }
	CActivationDesc GetDesc() const override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TInputBlobs; }

private:
	TCalculationMode mode = DefaultCalculationMode;

	// 1
	CFloatHandleVar oneVar;
	// 0.5
	CFloatHandleVar halfVar;
	// 1/sqrt(2)
	CFloatHandleVar sqrt2InvVar;
	// 1/sqrt(2pi)
	CFloatHandleVar sqrt2PiInvVar;
	// 1.702f
	CFloatHandleVar approxScaleVar;

	CPtr<CDnnBlob> erfMemoization;

	void runPrecise();
	void runFastApproximate();
	void backwardPrecise();
	void backwardFastApproximate();
};

NEOML_API CLayerWrapper<CGELULayer> Gelu();

//------------------------------------------------------------------------------------------------------------

// Creates an activation layer using the specified activation function
CPtr<CBaseLayer> NEOML_API CreateActivationLayer( IMathEngine& mathEngine, const CActivationDesc& activation );

void NEOML_API StoreActivationDesc( const CActivationDesc& desc, CArchive& archive );

CActivationDesc NEOML_API LoadActivationDesc( CArchive& archive );

} // namespace NeoML
