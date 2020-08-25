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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/DnnSolver.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

static CMap<CString, TCreateSolverFunction>& getRegisteredSolvers()
{
	static CMap<CString, TCreateSolverFunction> registeredSolvers;
	return registeredSolvers;
}

// Class name hash to compare type_info
struct CTypeInfoNameHash {
	static int HashKey( const std::type_info* key )
	{ 
		return GetMBCStringHash( key->name() );
	}

	static bool IsEqual( const std::type_info* first, const std::type_info* second )
	{
		return ( ::strcmp( first->name(), second->name() ) == 0 );
	}
};

static CMap<const std::type_info*, CString, CTypeInfoNameHash>& getSolverNames()
{
	static CMap<const std::type_info*, CString, CTypeInfoNameHash> solverNames;
	return solverNames;
}

void RegisterSolverName( const char* className, const std::type_info& typeInfo, TCreateSolverFunction function )
{
	NeoAssert( !getRegisteredSolvers().Has( className ) );
	getRegisteredSolvers().Add( className, function );
	getSolverNames().Add( &typeInfo, className );
}

void UnregisterSolverName( const std::type_info& typeInfo )
{
	getRegisteredSolvers().Delete( getSolverNames().Get( &typeInfo ) );
	getSolverNames().Delete( &typeInfo );
}

static CPtr<CDnnSolver> createSolver( IMathEngine& mathEngine, const CString& className )
{
	TMapPosition pos = getRegisteredSolvers().GetFirstPosition( className );
	if( pos == NotFound ) {
		return 0;
	}
	return getRegisteredSolvers().GetValue( pos )( mathEngine );
}

static CString getSolverName( CDnnSolver* solver )
{
	if( solver == 0 ) {
		return CString();
	}
	const std::type_info& solverType = typeid( *solver );
	TMapPosition pos = getSolverNames().GetFirstPosition( &solverType );
	if( pos == NotFound ) {
		return CString();
	}
	return getSolverNames().GetValue( pos );
}

void SerializeSolver( CArchive& archive, IMathEngine& mathEngine, CPtr<CDnnSolver>& solver )
{
	if( archive.IsStoring() ) {
		archive << getSolverName( solver );
		if( solver != 0 ) {
			solver->Serialize( archive );
		}
	} else if( archive.IsLoading() ) {
		CString name;
		archive >> name;
		solver = createSolver( mathEngine, name );
		if( solver != 0 ) {
			solver->Serialize( archive );
		}
	} else {
		NeoAssert( false );
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
REGISTER_NEOML_SOLVER( CDnnSimpleGradientSolver, "NeoMLDnnSimpleGradientSolver" )
REGISTER_NEOML_SOLVER( CDnnAdaptiveGradientSolver, "NeoMLDnnAdaptiveGradientSolver" )
REGISTER_NEOML_SOLVER( CDnnNesterovGradientSolver, "NeoMLDnnNesterovGradientSolver" )
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CDnnSolver::CDnnSolver( IMathEngine& _mathEngine ) :
	mathEngine( _mathEngine ),
	learningRate( 0.01f ),
	regularizationL2( 1e-4f ),
	regularizationL1( 0.f ),
	maxGradientNorm( -1.f )
{
}

// Calculates the layer parameter gradients to then use them in Train method
void CDnnSolver::AddDiff( CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramDiffBlobs )
{
	NeoAssert( layer != 0 );

	CDiffBlobSum& paramDiffBlobsSum = layerToParamDiffBlobsSum.GetOrCreateValue( layer->GetLayerId() );
	++paramDiffBlobsSum.Count;

	layerToPtr.GetOrCreateValue( layer->GetLayerId() ) = layer;

	if( paramDiffBlobsSum.Count == 1 ) {
		// The first term
		NeoAssert( paramDiffBlobsSum.Sum.IsEmpty() );
		paramDiffBlobs.CopyTo( paramDiffBlobsSum.Sum );
		return;
	}

	NeoAssert( paramDiffBlobsSum.Sum.Size() == paramDiffBlobs.Size() );
	for( int i = 0; i < paramDiffBlobs.Size(); i++ ) {
		paramDiffBlobsSum.Sum[i]->Add( paramDiffBlobs[i] );
	}
}

// Modifies the trainable parameters of the network layers, using the accumulated gradient values 
// and the history of previous modifications (moment, etc.)
void CDnnSolver::Train()
{
	OnTrain();

	CFloatHandleStackVar oneDivEpoch( mathEngine );

	for( TMapPosition pos = layerToParamDiffBlobsSum.GetFirstPosition(); pos != NotFound;
		pos = layerToParamDiffBlobsSum.GetNextPosition( pos ) )
	{
		const CString& layerId = layerToParamDiffBlobsSum.GetKey( pos );
		CDiffBlobSum& paramDiffBlobsSum = layerToParamDiffBlobsSum.GetValue( pos );
		if( paramDiffBlobsSum.Sum.IsEmpty() ) {
			continue;
		}
		NeoAssert( paramDiffBlobsSum.Count > 0 );

		// Take the average of the gradients to simulate that the elements from all runs were in the same batch
		// TODO: weighted average
		if( paramDiffBlobsSum.Count > 1 ) {
			oneDivEpoch.SetValue( 1.f / paramDiffBlobsSum.Count );
			for( int i = 0; i < paramDiffBlobsSum.Sum.Size(); i++ ) {
				MathEngine().VectorMultiply( paramDiffBlobsSum.Sum[i]->GetData(), paramDiffBlobsSum.Sum[i]->GetData(),
					paramDiffBlobsSum.Sum[i]->GetDataSize(), oneDivEpoch );
			}
		}

		clipGradients( paramDiffBlobsSum.Sum );

		// Train the layer based on the calculated diff data
		CBaseLayer* layer = layerToPtr.Get( layerId );
		TrainLayer( layer, layer->paramBlobs, paramDiffBlobsSum.Sum, layerToGradientHistory.GetOrCreateValue( layerId ) );

		// Clear the diff data
		paramDiffBlobsSum.Sum.Empty();
		paramDiffBlobsSum.Count = 0;
	}
}

void CDnnSolver::Reset()
{
	layerToParamDiffBlobsSum.DeleteAll();
	layerToGradientHistory.DeleteAll();
	layerToPtr.DeleteAll();
	OnReset();
}

void CDnnSolver::clipGradients(const CObjectArray<CDnnBlob>& paramDiffBlobs)
{
	if(maxGradientNorm < 0 || paramDiffBlobs.Size() == 0) {
		return;
	}

	// Calculate the parameter gradient norm
	CFloatHandleStackVar tempVar( MathEngine() );
	CFloatHandleStackVar gradVar( MathEngine() );
	MathEngine().VectorDotProduct(paramDiffBlobs[0]->GetData(), paramDiffBlobs[0]->GetData(),
		paramDiffBlobs[0]->GetDataSize(), gradVar.GetHandle());
	for(int i = 1; i < paramDiffBlobs.Size(); ++i) {
		MathEngine().VectorDotProduct(paramDiffBlobs[i]->GetData(), paramDiffBlobs[i]->GetData(),
			paramDiffBlobs[i]->GetDataSize(), tempVar.GetHandle());
		MathEngine().VectorAdd(gradVar.GetHandle(), tempVar.GetHandle(), gradVar.GetHandle(), 1);
	}
	MathEngine().VectorSqrt(gradVar.GetHandle(), gradVar.GetHandle(), 1);

	// Calculate scale
	tempVar.SetValue(maxGradientNorm);
	MathEngine().VectorEltwiseMax(gradVar.GetHandle(), tempVar.GetHandle(), gradVar.GetHandle(), 1);
	MathEngine().VectorEltwiseDivide(tempVar.GetHandle(), gradVar.GetHandle(), tempVar.GetHandle(), 1);

	// Decrease the gradient
	for(int i = 0; i < paramDiffBlobs.Size(); ++i) {
		MathEngine().VectorMultiply(paramDiffBlobs[i]->GetData(), paramDiffBlobs[i]->GetData(),
			paramDiffBlobs[i]->GetDataSize(), tempVar.GetHandle());
	}
}

static const int DnnSolverVersion = 1;

void CDnnSolver::Serialize( CArchive& archive )
{
	archive.SerializeVersion( DnnSolverVersion );
	if( archive.IsStoring() ) {
		archive << layerToParamDiffBlobsSum.Size();
		for( int pos = layerToParamDiffBlobsSum.GetFirstPosition(); pos != NotFound;
			pos = layerToParamDiffBlobsSum.GetNextPosition( pos ) )
		{
			archive << layerToParamDiffBlobsSum.GetKey( pos );
			archive << layerToParamDiffBlobsSum.GetValue( pos ).Count;
			SerializeBlobs( mathEngine, archive, layerToParamDiffBlobsSum.GetValue( pos ).Sum );
		}

		archive << layerToGradientHistory.Size();
		for( int pos = layerToGradientHistory.GetFirstPosition(); pos != NotFound;
			pos = layerToGradientHistory.GetNextPosition( pos ) )
		{
			archive << layerToGradientHistory.GetKey( pos );
			SerializeBlobs( mathEngine, archive, layerToGradientHistory.GetValue( pos ) );
		}
		archive << learningRate << regularizationL1 << regularizationL2 << maxGradientNorm;
	} else {
		layerToParamDiffBlobsSum.DeleteAll();
		layerToGradientHistory.DeleteAll();
		layerToPtr.DeleteAll();
		int size;
		archive >> size;
		for( int i = 0; i < size; ++i ) {
			CString layerId;
			archive >> layerId;
			CDiffBlobSum& blobSum = layerToParamDiffBlobsSum.GetOrCreateValue( layerId );
			archive >> blobSum.Count;
			SerializeBlobs( mathEngine, archive, blobSum.Sum );
		}

		archive >> size;
		for( int i = 0; i < size; ++i ) {
			CString layerId;
			archive >> layerId;
			SerializeBlobs( mathEngine, archive, layerToGradientHistory.GetOrCreateValue( layerId ) );
		}
		archive >> learningRate >> regularizationL1 >> regularizationL2 >> maxGradientNorm;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CDnnSimpleGradientSolver::CDnnSimpleGradientSolver( IMathEngine& mathEngine ) :
	CDnnSolver( mathEngine ),
	momentDecayRate( 0.9f ),
	isInCompatibilityMode( false ),
	tempVariables( CDnnBlob::CreateVector( mathEngine, CT_Float, TV_Count ) )
{
	SetLearningRate(0.01f);
	SetL2Regularization(1e-4f);
}

static const int DnnSimpleGradientSolverVersion = 1;

void CDnnSimpleGradientSolver::Serialize( CArchive& archive )
{
	archive.SerializeVersion( DnnSimpleGradientSolverVersion );
	CDnnSolver::Serialize( archive );
	archive.Serialize( momentDecayRate );
	archive.Serialize( isInCompatibilityMode );
}

void CDnnSimpleGradientSolver::TrainLayer( const CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramBlobs,
	const CObjectArray<CDnnBlob>& paramDiffBlobs, CObjectArray<CDnnBlob>& gradientHistory )
{
	if(gradientHistory.Size() == 0) {
		for (int i = 0; i < paramDiffBlobs.Size(); ++i) {
			CDnnBlob* blob = paramDiffBlobs[i]->GetClone();
			blob->Clear();
			gradientHistory.Add(blob);
		}
	}

	float rate = layer->GetBaseLearningRate() * GetLearningRate();
	float regL1 = layer->GetBaseL1RegularizationMult() * GetL1Regularization();
	float regL2 = layer->GetBaseL2RegularizationMult() * GetL2Regularization();

	// Set the values of the variables
	CFastArray<float, TV_Count> varValues;
	varValues.SetSize( TV_Count );

	varValues[TV_MomentDecayRateVar] = momentDecayRate;
	varValues[TV_OpMomentDecayRateVar] = 1 - momentDecayRate;
	varValues[TV_OpRegL2MomentDecayRateVar] = isInCompatibilityMode ? ( 1 - momentDecayRate ) * regL2 : -rate * regL2;
	varValues[TV_RateVar] = (-rate);
	varValues[TV_L1Threshold] = regL1;
	varValues[TV_L1Mult] = isInCompatibilityMode ? 1.f : -rate;

	MathEngine().DataExchangeTyped( tempVariables->GetData(), varValues.GetPtr(), TV_Count );

	for(int i = 0; i < paramBlobs.Size(); ++i) {
		int dataSize = paramBlobs[i]->GetDataSize();
		// Update the gradient in history
		MathEngine().VectorMultiply( gradientHistory[i]->GetData(),
			gradientHistory[i]->GetData(), dataSize, tempVariables->GetData( {TV_MomentDecayRateVar} ) );
		MathEngine().VectorMultiplyAndAdd( gradientHistory[i]->GetData(), paramDiffBlobs[i]->GetData(), 
			gradientHistory[i]->GetData(), dataSize,
			tempVariables->GetData( { isInCompatibilityMode ? TV_OpMomentDecayRateVar : TV_RateVar } ) );

		if(regL2 > 0) {
			MathEngine().VectorMultiplyAndAdd( gradientHistory[i]->GetData(), paramBlobs[i]->GetData(),
				gradientHistory[i]->GetData(), dataSize, tempVariables->GetData( {TV_OpRegL2MomentDecayRateVar} ) );
		}
		if(regL1 > 0) {
			MathEngine().VectorL1DiffAdd( gradientHistory[i]->GetData(), paramBlobs[i]->GetData(),
				gradientHistory[i]->GetData(), dataSize, tempVariables->GetData( {TV_L1Threshold} ),
					tempVariables->GetData( {TV_L1Mult} ) );
		}

		// Add regularization and gradient
		if( isInCompatibilityMode ) {
			MathEngine().VectorMultiplyAndAdd( paramBlobs[i]->GetData(), gradientHistory[i]->GetData(),
				paramBlobs[i]->GetData(), dataSize, tempVariables->GetData( { TV_RateVar } ) );
		} else {
			MathEngine().VectorAdd( paramBlobs[i]->GetData(), gradientHistory[i]->GetData(),
				paramBlobs[i]->GetData(), dataSize );
		}
	}
}

CDnnAdaptiveGradientSolver::CDnnAdaptiveGradientSolver( IMathEngine& mathEngine ) :
	CDnnSolver( mathEngine ),
	momentDecayRate(0.9f),
	momentDecayRateN(1.f),
	secondMomentDecayRate(0.99f),
	secondMomentDecayRateN(1.f),
	epsilon(1e-6f),
	isAmsGradEnabled( false ),
	isInCompatibilityMode( false ),
	tempVariables( CDnnBlob::CreateVector( mathEngine, CT_Float, TV_Count ) )
{
	SetLearningRate(0.01f);
	SetL2Regularization(1e-6f);
}

// Turns on the AMSGrad mode; you can call this method only before training
void CDnnAdaptiveGradientSolver::EnableAmsGrad( bool enable )
{
	Reset();
	isAmsGradEnabled = enable;
}

static const int DnnAdaptiveGradientSolver = 1;

void CDnnAdaptiveGradientSolver::Serialize( CArchive& archive )
{
	archive.SerializeVersion( DnnAdaptiveGradientSolver );
	CDnnSolver::Serialize( archive );
	archive.Serialize( momentDecayRate );
	archive.Serialize( momentDecayRateN );
	archive.Serialize( secondMomentDecayRate );
	archive.Serialize( secondMomentDecayRateN );
	archive.Serialize( epsilon );
	archive.Serialize( isAmsGradEnabled );
	archive.Serialize( isInCompatibilityMode );
}

void CDnnAdaptiveGradientSolver::OnReset()
{
	momentDecayRateN = 1.f;
	secondMomentDecayRateN = 1.f;
}

// Prepares for the next training iteration
void CDnnAdaptiveGradientSolver::OnTrain()
{
	// Update the solver parameters that depend on run number
	momentDecayRateN *= momentDecayRate;
	secondMomentDecayRateN *= secondMomentDecayRate;
}

void CDnnAdaptiveGradientSolver::TrainLayer( const CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramBlobs, 
	const CObjectArray<CDnnBlob>& paramDiffBlobs, CObjectArray<CDnnBlob>& gradientHistory )
{
	if(gradientHistory.Size() == 0) {
		// Create blobs
		int gradientHistoryTypeCount = IsAmsGradEnabled() ? GHTC_AmsGrad : GHTC_Default;
		for( int j = 0; j < gradientHistoryTypeCount; j++ ) {
			for(int i = 0; i < paramDiffBlobs.Size(); ++i) {
				CDnnBlob* blob = paramDiffBlobs[i]->GetClone();
				blob->Clear();
				gradientHistory.Add( blob );
			}
		}
	}

	// Add regularization and add diffs to parameters
	float rate = layer->GetBaseLearningRate() * GetLearningRate() * sqrtf(1 - secondMomentDecayRateN);
	if( !isInCompatibilityMode ) {
		rate /= (1 - momentDecayRateN);
	}

	float regL1 = layer->GetBaseL1RegularizationMult() * GetL1Regularization();
	float regL2 = layer->GetBaseL2RegularizationMult() * GetL2Regularization();

	// Set the values of the variables
	CFastArray<float, 9> varValues;
	varValues.SetSize( TV_Count );

	varValues[TV_MomentDecayRateVar] = momentDecayRate;
	varValues[TV_SecondMomentDecayRateVar] = secondMomentDecayRate;
	varValues[TV_RegL2Var] = regL2;
	varValues[TV_OpMomentDecayRateVar] = 1 - momentDecayRate;
	varValues[TV_OpSecondMomentDecayRateVar] = 1 - secondMomentDecayRate;
	varValues[TV_RateVar] = -rate;
	varValues[TV_L1Threshold] = regL1;
	varValues[TV_L1Mult] = 1.f;
	varValues[TV_EpsilonVar] = epsilon;

	MathEngine().DataExchangeTyped<float>( tempVariables->GetData(), varValues.GetPtr(), TV_Count );

	for(int i = 0; i < paramBlobs.Size(); ++i) {
		int dataSize = paramBlobs[i]->GetDataSize();
		CDnnBlob* moment = gradientHistory[i];
		CDnnBlob* secondMoment = gradientHistory[i + paramDiffBlobs.Size() * GHT_SecondMomentAverage];

		if(temporaryBlob == 0 || temporaryBlob->GetDataSize() < paramDiffBlobs[i]->GetDataSize()) {
			temporaryBlob = CDnnBlob::CreateVector( MathEngine(), CT_Float, paramDiffBlobs[i]->GetDataSize() );
		}

		CDnnBlob* paramDiffBlob = paramDiffBlobs[i];

		// Add regularization
		if(regL2 > 0) {
			MathEngine().VectorMultiplyAndAdd(paramDiffBlob->GetData(), paramBlobs[i]->GetData(),
				temporaryBlob->GetData(), dataSize, tempVariables->GetData( {TV_RegL2Var} ));
			paramDiffBlob = temporaryBlob;
		}
		if(regL1 > 0) {
			MathEngine().VectorL1DiffAdd(paramDiffBlob->GetData(), paramBlobs[i]->GetData(),
				temporaryBlob->GetData(), dataSize, tempVariables->GetData( {TV_L1Threshold} ),
				tempVariables->GetData( {TV_L1Mult} ));
			paramDiffBlob = temporaryBlob;
		}

		// Update the historical gradient
		MathEngine().VectorMultiply(moment->GetData(), moment->GetData(), dataSize,
			tempVariables->GetData( {TV_MomentDecayRateVar} ));
		MathEngine().VectorMultiplyAndAdd(moment->GetData(), paramDiffBlob->GetData(),
			moment->GetData(), dataSize, tempVariables->GetData( {TV_OpMomentDecayRateVar} ));
		// Calculate the historical average squared gradient
		MathEngine().VectorEltwiseMultiply(paramDiffBlob->GetData(), paramDiffBlob->GetData(),
			temporaryBlob->GetData(), dataSize);
		MathEngine().VectorMultiply(secondMoment->GetData(), secondMoment->GetData(), dataSize,
			tempVariables->GetData( {TV_SecondMomentDecayRateVar} ));
		MathEngine().VectorMultiplyAndAdd(secondMoment->GetData(), temporaryBlob->GetData(), 
			secondMoment->GetData(), dataSize, tempVariables->GetData( {TV_OpSecondMomentDecayRateVar} ));
		if( IsAmsGradEnabled() ) {
			// Update the maximum of the average
			CDnnBlob* secondMomentMaxAverage = gradientHistory[i + paramDiffBlobs.Size() * GHT_SecondMomentMaxAverage];
			MathEngine().VectorEltwiseMax( secondMomentMaxAverage->GetData(), secondMoment->GetData(),
				secondMomentMaxAverage->GetData(), secondMomentMaxAverage->GetDataSize() );
			// Calculate the square root of the historical maximum of average squared gradient
			MathEngine().VectorSqrt( secondMomentMaxAverage->GetData(), temporaryBlob->GetData(), dataSize );
		} else {
			// Calculate the square root of the historical average squared gradient
			MathEngine().VectorSqrt( secondMoment->GetData(), temporaryBlob->GetData(), dataSize );
		}
		// Add epsilon before dividing
		MathEngine().VectorAddValue(temporaryBlob->GetData(), temporaryBlob->GetData(), dataSize,
			tempVariables->GetData( {TV_EpsilonVar} ));
		// Divide the historical gradient by the square root
		MathEngine().VectorEltwiseDivide(moment->GetData(), temporaryBlob->GetData(), 
			temporaryBlob->GetData(), dataSize);
		// Add the gradient
		MathEngine().VectorMultiplyAndAdd(paramBlobs[i]->GetData(), temporaryBlob->GetData(),
			paramBlobs[i]->GetData(), dataSize, tempVariables->GetData( {TV_RateVar} ));
	}
}

CDnnNesterovGradientSolver::CDnnNesterovGradientSolver( IMathEngine& mathEngine ) :
	CDnnSolver( mathEngine ),
	momentDecayRate( 0.9f ),
	secondMomentDecayRate( 0.99f ),
	secondMomentDecayRateN( 1.f ),
	epsilon( 1e-6f ),
	isAmsGradEnabled( false ),
	trainCount( 0 ),
	productMuT( 1.f ),
	tempVariables( CDnnBlob::CreateVector( mathEngine, CT_Float, TV_Count ) )
{
	SetLearningRate( 0.01f );
	SetL2Regularization( 1e-6f );
}

// Turns on the AMSGrad mode. The solver will be reset to initial state
void CDnnNesterovGradientSolver::EnableAmsGrad( bool enable )
{
	Reset();
	isAmsGradEnabled = enable;
}

static const int DnnNesterovGradientSolverVersion = 1;

void CDnnNesterovGradientSolver::Serialize( CArchive& archive )
{
	archive.SerializeVersion( DnnNesterovGradientSolverVersion );
	CDnnSolver::Serialize( archive );
	archive.Serialize( momentDecayRate );
	archive.Serialize( secondMomentDecayRate );
	archive.Serialize( secondMomentDecayRateN );
	archive.Serialize( epsilon );
	archive.Serialize( isAmsGradEnabled );
	archive.Serialize( trainCount );
	archive.Serialize( productMuT );
}

void CDnnNesterovGradientSolver::OnReset()
{
	trainCount = 0;
	secondMomentDecayRateN = 1.f;
	productMuT = 1.f;
}

// Prepares for the next training iteration
void CDnnNesterovGradientSolver::OnTrain()
{
	// Update the solver parameters that depend on run number
	secondMomentDecayRateN *= secondMomentDecayRate;
	// The "magic numbers" are from the reference paper
	trainCount++;
	muT = momentDecayRate * ( 1 - 0.5f * powf( 0.96f, trainCount * 0.004f ) );
	muTPlusOne = momentDecayRate * ( 1 - 0.5f * powf( 0.96f, ( trainCount + 1 ) * 0.004f ) );
	productMuT *= muT;
}

void CDnnNesterovGradientSolver::TrainLayer( const CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramBlobs,
	const CObjectArray<CDnnBlob>& paramDiffBlobs, CObjectArray<CDnnBlob>& gradientHistory )
{
	if(gradientHistory.Size() == 0) {
		// Create blobs
		int gradientHistoryTypeCount = IsAmsGradEnabled() ? GHTC_AmsGrad : GHTC_Default;
		for( int j = 0; j < gradientHistoryTypeCount; j++ ) {
			for(int i = 0; i < paramDiffBlobs.Size(); ++i) {
				CDnnBlob* blob = paramDiffBlobs[i]->GetClone();
				blob->Clear();
				gradientHistory.Add( blob );
			}
		}
	}

	// Apply regularization and add diffs to the parameters
	float rate = layer->GetBaseLearningRate() * GetLearningRate();
	float regL1 = layer->GetBaseL1RegularizationMult() * GetL1Regularization();
	float regL2 = layer->GetBaseL2RegularizationMult() * GetL2Regularization();

	// Set the values for the variables
	CFastArray<float, TV_Count> varValues;
	varValues.SetSize( TV_Count );

	varValues[TV_MomentDecayRateVar] = momentDecayRate;
	varValues[TV_SecondMomentDecayRateVar] = secondMomentDecayRate;
	varValues[TV_RegL2Var] = regL2;
	varValues[TV_OpMomentDecayRateVar] = 1 - momentDecayRate;
	varValues[TV_OpSecondMomentDecayRateVar] = 1 - secondMomentDecayRate;
	varValues[TV_RateVar] = -rate;
	varValues[TV_L1Threshold] = regL1;
	varValues[TV_L1Mult] = 1.f;
	varValues[TV_EpsilonVar] = epsilon;
	varValues[TV_MBarGradMultVar] = ( 1.f - muT ) / ( 1.f - productMuT );
	varValues[TV_MBarMomentMultVar] = muTPlusOne / ( 1.f - productMuT * muTPlusOne );
	varValues[TV_InvOpSecondMomentDecayRateNVar] = 1 / ( 1 - secondMomentDecayRateN );

	MathEngine().DataExchangeTyped( tempVariables->GetData(), varValues.GetPtr(), TV_Count );

	for( int i = 0; i < paramBlobs.Size(); ++i ) {
		int dataSize = paramBlobs[i]->GetDataSize();
		CDnnBlob* moment = gradientHistory[i];
		CDnnBlob* secondMoment = gradientHistory[i + paramDiffBlobs.Size() * GHT_SecondMomentAverage];

		if( temporaryBlob == 0 || temporaryBlob->GetDataSize() < paramDiffBlobs[i]->GetDataSize() ) {
			temporaryBlob = CDnnBlob::CreateVector( MathEngine(), CT_Float, paramDiffBlobs[i]->GetDataSize() );
			mBarBlob = temporaryBlob->GetClone();
		}

		CDnnBlob* paramDiffBlob = paramDiffBlobs[i];

		// Add regularization
		if( regL2 > 0 ) {
			MathEngine().VectorMultiplyAndAdd( paramDiffBlob->GetData(), paramBlobs[i]->GetData(),
				temporaryBlob->GetData(), dataSize, tempVariables->GetData( {TV_RegL2Var} ) );
			paramDiffBlob = temporaryBlob;
		}
		if( regL1 > 0 ) {
			MathEngine().VectorL1DiffAdd( paramDiffBlob->GetData(), paramBlobs[i]->GetData(),
				temporaryBlob->GetData(), dataSize, tempVariables->GetData( {TV_L1Threshold} ),
				tempVariables->GetData( {TV_L1Mult} ) );
			paramDiffBlob = temporaryBlob;
		}

		// Update the historical gradient
		MathEngine().VectorMultiply( moment->GetData(), moment->GetData(), dataSize,
			tempVariables->GetData( {TV_MomentDecayRateVar}) );
		MathEngine().VectorMultiplyAndAdd( moment->GetData(), paramDiffBlob->GetData(),
			moment->GetData(), dataSize, tempVariables->GetData( {TV_OpMomentDecayRateVar} ) );
		// Calculate the historical average squared gradient
		MathEngine().VectorEltwiseMultiply( paramDiffBlob->GetData(), paramDiffBlob->GetData(),
			temporaryBlob->GetData(), dataSize );
		MathEngine().VectorMultiply( secondMoment->GetData(), secondMoment->GetData(), dataSize,
			tempVariables->GetData( {TV_SecondMomentDecayRateVar} ) );
		MathEngine().VectorMultiplyAndAdd( secondMoment->GetData(), temporaryBlob->GetData(),
			secondMoment->GetData(), dataSize, tempVariables->GetData( {TV_OpSecondMomentDecayRateVar} ) );

		// Calculate the auxiliary variables (notations taken from the reference paper)
		// m with a dash
		CFloatHandle mBar = mBarBlob->GetData();
		MathEngine().VectorMultiply( paramDiffBlob->GetData(), mBar, dataSize,
			tempVariables->GetData( {TV_MBarGradMultVar} ) );
		MathEngine().VectorMultiplyAndAdd( mBar, moment->GetData(), mBar, dataSize,
			tempVariables->GetData( {TV_MBarMomentMultVar} ) );

		// sqrt(n with a hat) + eps
		if( IsAmsGradEnabled() ) {
			// Update the maximum average
			CDnnBlob* secondMomentMaxAverage = gradientHistory[i + paramDiffBlobs.Size() * GHT_SecondMomentMaxAverage];
			MathEngine().VectorEltwiseMax( secondMomentMaxAverage->GetData(), secondMoment->GetData(),
				secondMomentMaxAverage->GetData(), secondMomentMaxAverage->GetDataSize() );
			// n with a hat calculated for the maximum of the second moment moving mean
			MathEngine().VectorMultiply( secondMomentMaxAverage->GetData(), temporaryBlob->GetData(), dataSize, 
				tempVariables->GetData( { TV_InvOpSecondMomentDecayRateNVar } ) );
		} else {
			// n with a hat calculated for the second momentum moving mean
			MathEngine().VectorMultiply( secondMoment->GetData(), temporaryBlob->GetData(), dataSize,
				tempVariables->GetData( { TV_InvOpSecondMomentDecayRateNVar } ) );
		}
		MathEngine().VectorSqrt( temporaryBlob->GetData(), temporaryBlob->GetData(), dataSize );
		MathEngine().VectorAddValue( temporaryBlob->GetData(), temporaryBlob->GetData(), dataSize,
			tempVariables->GetData( { TV_EpsilonVar } ) );
		// Calculate the final diff
		MathEngine().VectorEltwiseDivide( mBar, temporaryBlob->GetData(), temporaryBlob->GetData(), dataSize );
		// Update parameters
		MathEngine().VectorMultiplyAndAdd( paramBlobs[i]->GetData(), temporaryBlob->GetData(),
			paramBlobs[i]->GetData(), dataSize, tempVariables->GetData( {TV_RateVar} ) );
	}
}

} // namespace NeoML
