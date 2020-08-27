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
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

class CDnnBlob;
class CBaseLayer;
class CDnn;

// The base optimizer class
class NEOML_API CDnnSolver : virtual public IObject {
public:
	// Stores the calculated values of layer parameters gradients for further use in Train method
	void AddDiff( CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramDiffBlobs );

	// Modifies the trainable parameters of the network layers, 
	// using the accumulated gradients and previous steps' history (moment, etc.) 
	void Train();

	// Resets to the initial state
	void Reset();

	// Learning rate
	float GetLearningRate() const { return learningRate; }
	void SetLearningRate( float _learningRate ) { learningRate = _learningRate; }
	// Regularization
	float GetL2Regularization() const { return regularizationL2; }
	void SetL2Regularization( float _regularization ) { regularizationL2 = _regularization; }
	float GetL1Regularization() const { return regularizationL1; }
	void SetL1Regularization(float _regularization) { regularizationL1 = _regularization; }
	// Upper limit for gradient norm (if set to < 0, that means no limit)
	float GetMaxGradientNorm() const { return maxGradientNorm; }
	void SetMaxGradientNorm(float _maxGradientNorm) { maxGradientNorm = _maxGradientNorm; }

	// Serialize to archive
	virtual void Serialize( CArchive& archive, CDnn& dnn );

protected:
	explicit CDnnSolver( IMathEngine& mathEngine );

	// Gets the reference to the math engine
	IMathEngine& MathEngine() const { return mathEngine; }

	// Called once on Reset method call
	// Resets the stats in the inheriting instances to the initial state
	virtual void OnReset() {}

	// On each training step the method is called once, before the call to TrainLayer for all layers
	virtual void OnTrain() {}

	// Modifies trainable parameters of a given layer, applying the paramDiffBlobs differences 
	// and using the learningHistory stored history
	// learningHistory may change during training
	virtual void TrainLayer( const CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramBlobs,
		const CObjectArray<CDnnBlob>& paramDiffBlobs, CObjectArray<CDnnBlob>& learningHistory ) = 0;

private:
	IMathEngine& mathEngine;
	float learningRate;
	float regularizationL2;
	float regularizationL1;
	float maxGradientNorm;

	// The blobs sum
	struct CDiffBlobSum {
		CDiffBlobSum() : Count( 0 ) {}

		CObjectArray<CDnnBlob> Sum; // the blobs sums
		int Count; // the number of terms in each sum
	};

	// The buffers used to add up the gradients from several AddDiff calls
	CMap<CBaseLayer*, CDiffBlobSum> layerToParamDiffBlobsSum;
	// The buffers for storing gradients history and moment
	// Used in the inheriting classes
	CMap<CBaseLayer*, CObjectArray<CDnnBlob>> layerToGradientHistory;

	// Clips gradients according to the settings
	void clipGradients(const CObjectArray<CDnnBlob>& paramDiffBlobs);

	// Telling the compiler that we intentionally using two-parameter Serialize instead of one declared in IObject
	using IObject::Serialize;
};

////////////////////////////////////////////////////////////////////////////////////////////////

// The macros for the internal name of a NeoML solver
// If this macros is used when declaring a class, that class may be registered as a NeoML solver
#define NEOML_DNN_SOLVER( className ) friend class CSolverClassRegistrar< className >;

// Registers the class as a NeoML solver
#define REGISTER_NEOML_SOLVER( classType, name ) static CSolverClassRegistrar< classType > __merge__1( _RegisterSolver, __LINE__ )( name );

typedef CPtr<CDnnSolver> ( *TCreateSolverFunction )( IMathEngine& mathEngine );

void NEOML_API RegisterSolverName( const char* className, const std::type_info& typeInfo, TCreateSolverFunction function );

void NEOML_API UnregisterSolverName( const std::type_info& typeInfo );

void NEOML_API SerializeSolver( CArchive& archive, CDnn& dnn, CPtr<CDnnSolver>& solver);

////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
class CSolverClassRegistrar {
public:
	explicit CSolverClassRegistrar( const char* solverName );
	~CSolverClassRegistrar();

private:
	static CPtr<CDnnSolver> createObject( IMathEngine& mathEngine ) { return FINE_DEBUG_NEW T( mathEngine ); }
};

template<class T>
inline CSolverClassRegistrar<T>::CSolverClassRegistrar( const char* solverName )
{
	RegisterSolverName( solverName, typeid( T ), createObject );
}

template<class T>
inline CSolverClassRegistrar<T>::~CSolverClassRegistrar()
{
	UnregisterSolverName( typeid( T ) );
}

////////////////////////////////////////////////////////////////////////////////////////////////

// Stochastic gradient descent with moment
class NEOML_API CDnnSimpleGradientSolver : public CDnnSolver {
	NEOML_DNN_SOLVER( CDnnSimpleGradientSolver )
public:
	CDnnSimpleGradientSolver( IMathEngine& mathEngine );

	// Moment decay rate (moment is a weighted sum of previous gradients)
	float GetMomentDecayRate() const { return momentDecayRate; }
	void SetMomentDecayRate(float decayRate) { momentDecayRate = decayRate; }

	bool IsInCompatibilityMode() const { return isInCompatibilityMode; }
	void SetCompatibilityMode( bool compatibilityMode ) { isInCompatibilityMode = compatibilityMode; }

	void Serialize( CArchive& archive, CDnn& dnn ) override;

protected:
	void TrainLayer( const CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramBlobs, 
		const CObjectArray<CDnnBlob>& paramDiffBlobs, CObjectArray<CDnnBlob>& gradientHistory ) override;

private:
	// Moment decay rate (moment is a weighted sum of previous gradients)
	float momentDecayRate;

	// Backward compatibility mode
	bool isInCompatibilityMode;

	// Temporary variables of Handle type, used for calculations
	enum TTempVariable {
		TV_MomentDecayRateVar = 0,
		TV_OpMomentDecayRateVar,
		TV_OpRegL2MomentDecayRateVar,
		TV_RateVar,
		TV_L1Threshold,
		TV_L1Mult,
		TV_Count
	};

	CPtr<CDnnBlob> tempVariables;
};

////////////////////////////////////////////////////////////////////////////////////////////////

// Stochastic gradient descent with moment and adapting stride for each coordinate
class NEOML_API CDnnAdaptiveGradientSolver : public CDnnSolver {
	NEOML_DNN_SOLVER( CDnnAdaptiveGradientSolver )
public:
	CDnnAdaptiveGradientSolver( IMathEngine& mathEngine );

	// Retrieves and sets the moment decay rate (moment is a weighted sum of previous gradients)
	float GetMomentDecayRate() const { return momentDecayRate; }
	void SetMomentDecayRate(float decayRate) { momentDecayRate = decayRate; }
	// Retrieves and sets the decay rate for the weighted sum of previous gradients, squared (aka second moment)
	float GetSecondMomentDecayRate() const { return secondMomentDecayRate; }
	void SetSecondMomentDecayRate(float decayRate) { secondMomentDecayRate = decayRate; }

	// Retrieves and sets the espilon used to avoid division by zero when calculating second moment
	float GetEpsilon() const { return epsilon; }
	void SetEpsilon( float newEpsilon ) { epsilon = newEpsilon; }

	bool IsInCompatibilityMode() const { return isInCompatibilityMode; }
	void SetCompatibilityMode( bool compatibilityMode ) { isInCompatibilityMode = compatibilityMode; }

	// AMSGrad helps against divergence and rapid vanishing of previous states memory, 
	// which may become a problem for the optimizers that use the moving mean for squared gradient history (Adam, NAdam, RMSprop).
	// (see https://openreview.net/pdf?id=ryQu7f-RZ)
	bool IsAmsGradEnabled() const { return isAmsGradEnabled; }
	// Turns AMSGrad mode on. May be called only before training starts.
	void EnableAmsGrad( bool enable );

	void Serialize( CArchive& archive, CDnn& dnn ) override;

protected:
	// Resets to the initial state
	void OnReset() override;
	// Prepares for the next training step
	void OnTrain() override;
	// Updates the trainable weights of the layer
	virtual void TrainLayer( const CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramBlobs,
		const CObjectArray<CDnnBlob>& paramDiffBlobs, CObjectArray<CDnnBlob>& gradientHistory ) override;

private:
	// The gradientHistory array stores the previous values of gradients of different types
	enum TGradientHistoryType {
		// First moment (moving mean)
		GHT_FirstMomentAverage,
		// Second moment (moving mean)
		GHT_SecondMomentAverage,
		// Maximum moving mean of the second moment (used for AMSGrad)
		GHT_SecondMomentMaxAverage,
	};
	// The number of gradient history types for different processing modes
	enum TGradientHistoryTypeCount {
		// With AMSGrad off
		GHTC_Default = 2,
		// With AMSGrad on
		GHTC_AmsGrad = 3
	};

	// Moment decay rate
	float momentDecayRate;
	// Moment decay rate raised to the power of N (the number of training steps)
	float momentDecayRateN;
	// Second moment decay rate
	float secondMomentDecayRate;
	// Second moment decay rate raised to the power of N (the number of training steps)
	float secondMomentDecayRateN;
	// The initial correction so there would be no division by zero
	float epsilon;
	// Indicates if AMSGrad is used
	bool isAmsGradEnabled;

	// Backward compatibility mode
	bool isInCompatibilityMode;

	enum TTempVariable {
		TV_MomentDecayRateVar = 0,
		TV_SecondMomentDecayRateVar,
		TV_RegL2Var,
		TV_OpMomentDecayRateVar,
		TV_OpSecondMomentDecayRateVar,
		TV_RateVar,
		TV_L1Threshold,
		TV_L1Mult,
		TV_EpsilonVar,
		TV_Count
	};

	// Temporary Handle variables for calculations
	CPtr<CDnnBlob> tempVariables;

	CPtr<CDnnBlob> temporaryBlob;
};

////////////////////////////////////////////////////////////////////////////////////////////////

// The optimizer that uses Nesterov moment
// http://cs229.stanford.edu/proj2015/054_report.pdf (Algo 8).
class NEOML_API CDnnNesterovGradientSolver : public CDnnSolver {
	NEOML_DNN_SOLVER( CDnnNesterovGradientSolver )
public:
	explicit CDnnNesterovGradientSolver( IMathEngine& mathEngine );

	// Retrieves and sets the moment decay rate (moment is a weighted sum of previous gradients)
	float GetMomentDecayRate() const { return momentDecayRate; }
	void SetMomentDecayRate( float decayRate ) { momentDecayRate = decayRate; }
	// Retrieves and sets the decay rate for the weighted sum of squares of previous gradients
	float GetSecondMomentDecayRate() const { return secondMomentDecayRate; }
	void SetSecondMomentDecayRate( float decayRate ) { secondMomentDecayRate = decayRate; }

	// Retrieves and sets the espilon used to avoid division by zero when calculating second moment
	float GetEpsilon() const { return epsilon; }
	void SetEpsilon( float newEpsilon ) { epsilon = newEpsilon; }

	// AMSGrad helps against divergence and rapid vanishing of previous states memory, 
	// which may become a problem for the optimizers that use the moving mean for squared gradient history (Adam, NAdam, RMSprop).
	// (see https://openreview.net/pdf?id=ryQu7f-RZ)
	bool IsAmsGradEnabled() const { return isAmsGradEnabled; }
	// Turns on AMSGrad mode. The algorithm is reset to initial state
	void EnableAmsGrad( bool enable );

	void Serialize( CArchive& archive, CDnn& dnn ) override;

protected:
	// Resets to the initial state
	void OnReset() override;
	// Prepares for the next training step
	void OnTrain() override;
	// Updates the trainable weights of the layer
	virtual void TrainLayer( const CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramBlobs,
		const CObjectArray<CDnnBlob>& paramDiffBlobs, CObjectArray<CDnnBlob>& gradientHistory ) override;

private:
	// The gradientHistory array stores the previous values of gradients of different types
	enum TGradientHistoryType {
		// First moment (moving mean)
		GHT_FirstMomentAverage,
		// Second moment (moving mean)
		GHT_SecondMomentAverage,
		// Maximum moving mean of the second moment (used for AMSGrad)
		GHT_SecondMomentMaxAverage,
	};
	// The number of gradient history types for different processing modes
	enum TGradientHistoryTypeCount {
		// With AMSGrad off
		GHTC_Default = 2,
		// With AMSGrad on
		GHTC_AmsGrad = 3
	};
	// Moment decay rate
	float momentDecayRate;
	// Second moment decay rate
	float secondMomentDecayRate;
	// Second moment decay rate raised to the power of N (the number of training steps)
	float secondMomentDecayRateN;
	// The initial correction so there would be no division by zero:
	float epsilon;
	// Indicates if AMSGrad is used
	bool isAmsGradEnabled;
	
	// Coefficients for moment schedule
	int trainCount; // the number of calls to Train
	float muT; // the mu coefficient for the current step
	float muTPlusOne; // the mu coefficient for the next step
	float productMuT; // the product of mu coefficient over all steps including the current one

	enum TTempVariable {
		TV_MomentDecayRateVar = 0,
		TV_SecondMomentDecayRateVar,
		TV_RegL2Var,
		TV_OpMomentDecayRateVar,
		TV_OpSecondMomentDecayRateVar,
		TV_RateVar,
		TV_L1Threshold,
		TV_L1Mult,
		TV_EpsilonVar,
		TV_InvOpSecondMomentDecayRateNVar, // 1 / (1 - secondMomentDecay ^ N)
		TV_MBarGradMultVar, // the gradient coefficient in the total sum
		TV_MBarMomentMultVar, // the moment coefficient in the total sum
		TV_Count
	};

	// Temporary blobs for calculations
	CPtr<CDnnBlob> tempVariables;

	CPtr<CDnnBlob> temporaryBlob;
	// m with a stroke (from the paper referred to)
	// It is a weighted sum of the gradient and the first moment
	CPtr<CDnnBlob> mBarBlob;
};

////////////////////////////////////////////////////////////////////////////////////////////////

// Lamb optimizer.
// https://arxiv.org/pdf/1904.00962.pdf
// Basic differences compared to Adam:
// 1) It doesn't use debiasing ( 1/(1 - beta^N)) for compensation of lack of history
//	Instead use warmup stage - linear increase of learning rate up to required value
// 2) It uses trustRatio multiplier - ratio between weights' norm and update vector's norm.
//	The idea is when update is huge (in comparison with weights), it slows down training speed
//	and as a result makes it more stable.
// 3) It uses non-modified L2-regularization (aka weight decay)
class NEOML_API CDnnLambGradientSolver : public CDnnSolver {
	NEOML_DNN_SOLVER( CDnnLambGradientSolver )
public:
	explicit CDnnLambGradientSolver( IMathEngine& mathEngine );

	// Match type used when checking if layer is excluded from weightDecay
	enum TExcludeLayerNameMatchType {
		// Exact match with given string
		ELNMT_Exact,
		// If layer's name contains given string
		ELNMT_Include
	};

	// Exclude layer from weightDecay optimization
	// layerName - layer name (or substring if ELMNT_Include is used)
	// type - match type
	// paramIndex - index of optimized layer parameter to exclude (NotFound means all layer parameters)
	void ExcludeWeightDecayLayer( const char* layerName, TExcludeLayerNameMatchType type = ELNMT_Exact,
		int paramIndex = NotFound );

	// Gets/set moment decay rate (weighted sum of previous gradients)
	// By default is equal to 0.9 
	float GetMomentDecayRate() const { return momentDecayRate; }
	void SetMomentDecayRate( float decayRate ) { momentDecayRate = decayRate; }
	// Gets/set second moment decay rate (weighted sum of squares of previous gradients)
	// By default is equal to 0.999
	float GetSecondMomentDecayRate() const { return secondMomentDecayRate; }
	void SetSecondMomentDecayRate( float decayRate ) { secondMomentDecayRate = decayRate; }

	// Gets/sets epsilon uwhich is sed to avoid division by zero during calculation
	// By default is equal to 1e-6
	float GetEpsilon() const { return epsilon; }
	void SetEpsilon( float newEpsilon ) { epsilon = newEpsilon; }

	// Max value for weight norm during WeightDecay
	// Negative value means no clip
	// By default is equal to -1
	float GetWeightDecayClip() { return weightDecayClip; }
	void SetWeightDecayClip( float value ) { weightDecayClip = value; }

	// Use normalizing multipier
	// By default is true.
	bool UseTrustRatio() const { return useTrustRatio; }
	void SetUseTrustRatio( bool value ) { useTrustRatio = value; }

	// Use NVLamb modification
	// https://medium.com/nvidia-ai/a-guide-to-optimizer-implementation-for-bert-at-scale-8338cc7f45fd
	// By default is false
	// The key difference is that gradient is normalized in the next way:
	//	g_i_j(n) = g_i_j(n) * 1.0 / max( 1.0, L2_norm(g(n-1)) )
	// where L2_norm(g) - L2 norm of the gradient of all the layers in the model
	// g(n-1) - is used because of NeoML API restrictions
	bool GetUseNVLamb() const { return useNvLamb; }
	void SetUseNVLamb( bool value ) { useNvLamb = value; }

	void Serialize( CArchive& archive, CDnn& dnn ) override;

protected:
	void TrainLayer( const CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramBlobs,
		const CObjectArray<CDnnBlob>& paramDiffBlobs, CObjectArray<CDnnBlob>& gradientHistory ) override;

	void OnTrain() override;

private:
	// The gradientHistory array stores the previous values of gradients of different types
	enum TGradientHistoryType {
		// First moment (moving mean)
		GHT_FirstMomentAverage,
		// Second moment (moving mean)
		GHT_SecondMomentAverage,
	};

	// Moment decay rate
	float momentDecayRate;
	// Second moment decay rate
	float secondMomentDecayRate;
	// The initial correction so there would be no division by zero:
	float epsilon;
	// Max weight norm during WeightDecay
	float weightDecayClip;
	// Is LAMB normalization used
	bool useTrustRatio;
	// Is NVLamb modification used
	bool useNvLamb;

	enum TTempVariable {
		TV_MomentDecayRateVar,
		TV_SecondMomentDecayRateVar,
		TV_OpMomentDecayRateVar,
		TV_OpSecondMomentDecayRateVar,
		TV_RateVar,
		TV_EpsilonVar,
		TV_WeightDecayVar,
		TV_ClipMultiplierVar,
		TV_LayerNormVar,
		TV_TrustRatioVar,
		TV_L2NormVar,

		TV_Count
	};

	CPtr<CDnnBlob> tempVariables;

	CPtr<CDnnBlob> tempBlob;

	CArray<float> layersGradientNormSquare;
	float totalGradientNorm;

	// Layer excluded from optimization
	struct CExcludedLayer {
		// Layer name (or substring)
		CString LayerName;
		// Match type (exact or substring)
		TExcludeLayerNameMatchType MatchType;
		// Parameter number
		// -1 if all parameters
		int ParamIndex;

		CExcludedLayer() : MatchType( ELNMT_Exact ), ParamIndex( NotFound ) {}
	};
	// Layers excluded from weight decay
	CArray<CExcludedLayer> excludedLayers;

	float calcL2Norm( const CConstFloatHandle& data, int dataSize ) const;
	void getWeightDecayIndices( const CBaseLayer& layer, int paramsCount, CHashTable<int>& indexes ) const;

	void calcNormalizeMultiplier( const CDnnBlob& weights, const CDnnBlob& update, const CFloatHandle& multiplier ) const;
};

} // namespace NeoML
