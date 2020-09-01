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
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

static CMap<CString, TCreateSolverFunction, CDefaultHash<CString>, RuntimeHeap>& getRegisteredSolvers()
{
	static CMap<CString, TCreateSolverFunction, CDefaultHash<CString>, RuntimeHeap> registeredSolvers;
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

static CMap<const std::type_info*, CString, CTypeInfoNameHash, RuntimeHeap>& getSolverNames()
{
	static CMap<const std::type_info*, CString, CTypeInfoNameHash, RuntimeHeap> solverNames;
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

void SerializeSolver( CArchive& archive, CDnn& dnn, CPtr<CDnnSolver>& solver )
{
	if( archive.IsStoring() ) {
		archive << getSolverName( solver );
		if( solver != 0 ) {
			solver->Serialize( archive, dnn );
		}
	} else if( archive.IsLoading() ) {
		CString name;
		archive >> name;
		solver = createSolver( dnn.GetMathEngine(), name );
		if( solver != 0 ) {
			solver->Serialize( archive, dnn );
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
REGISTER_NEOML_SOLVER( CDnnLambGradientSolver, "NeoMLDnnLambGradientSolver" )
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility functions for serialization

void mapLayerIdToPtr( CDnnLayerGraph& dnn, CMap<CString, CBaseLayer*>& result, const CString& prefix = "" )
{
	CArray<const char*> layerNames;
	dnn.GetLayerList( layerNames );
	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CPtr<CBaseLayer> layer = dnn.GetLayer( layerNames[layerIndex] );
		result.Add( prefix + layer->GetName(), layer.Ptr() );
		CCompositeLayer* compositePtr = dynamic_cast<CCompositeLayer*>( layer.Ptr() );
		if( compositePtr != nullptr ) {
			mapLayerIdToPtr( *compositePtr, result, prefix + compositePtr->GetName() );
		}
	}
}

void mapLayerPtrToId( CDnnLayerGraph& dnn, CMap<CBaseLayer*, CString>& result, const CString& prefix = "" )
{
	CArray<const char*> layerNames;
	dnn.GetLayerList( layerNames );
	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CPtr<CBaseLayer> layer = dnn.GetLayer( layerNames[layerIndex] );
		result.Add( layer.Ptr(), prefix + layer->GetName() );
		CCompositeLayer* compositePtr = dynamic_cast<CCompositeLayer*>( layer.Ptr() );
		if( compositePtr != nullptr ) {
			mapLayerPtrToId( *compositePtr, result, prefix + compositePtr->GetName() );
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CDnnSolver::CDnnSolver( IMathEngine& _mathEngine ) :
	mathEngine( _mathEngine ),
	learningRate( 0.01f ),
	regularizationL2( 0.f ),
	regularizationL1( 0.f ),
	maxGradientNorm( -1.f )
{
}

// Calculates the layer parameter gradients to then use them in Train method
void CDnnSolver::AddDiff( CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramDiffBlobs )
{
	NeoAssert( layer != 0 );

	CDiffBlobSum& paramDiffBlobsSum = layerToParamDiffBlobsSum.GetOrCreateValue( layer );
	++paramDiffBlobsSum.Count;

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
		CBaseLayer* layer = layerToParamDiffBlobsSum.GetKey( pos );
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
		TrainLayer( layer, layer->paramBlobs, paramDiffBlobsSum.Sum, layerToGradientHistory.GetOrCreateValue( layer ) );

		// Clear the diff data
		paramDiffBlobsSum.Sum.Empty();
		paramDiffBlobsSum.Count = 0;
	}
}

void CDnnSolver::Reset()
{
	layerToParamDiffBlobsSum.DeleteAll();
	layerToGradientHistory.DeleteAll();
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

static const int DnnSolverVersion = 0;

void CDnnSolver::Serialize( CArchive& archive, CDnn& dnn )
{
	archive.SerializeVersion( DnnSolverVersion );
	if( archive.IsStoring() ) {
		CMap<CBaseLayer*, CString> layerPtrToId;
		mapLayerPtrToId( dnn, layerPtrToId );

		archive << layerToParamDiffBlobsSum.Size();
		for( int pos = layerToParamDiffBlobsSum.GetFirstPosition(); pos != NotFound;
			pos = layerToParamDiffBlobsSum.GetNextPosition( pos ) )
		{
			archive << layerPtrToId[layerToParamDiffBlobsSum.GetKey( pos )];
			archive << layerToParamDiffBlobsSum.GetValue( pos ).Count;
			SerializeBlobs( mathEngine, archive, layerToParamDiffBlobsSum.GetValue( pos ).Sum );
		}

		archive << layerToGradientHistory.Size();
		for( int pos = layerToGradientHistory.GetFirstPosition(); pos != NotFound;
			pos = layerToGradientHistory.GetNextPosition( pos ) )
		{
			archive << layerPtrToId[layerToGradientHistory.GetKey( pos )];
			SerializeBlobs( mathEngine, archive, layerToGradientHistory.GetValue( pos ) );
		}
		archive << learningRate << regularizationL1 << regularizationL2 << maxGradientNorm;
	} else {
		CMap<CString, CBaseLayer*> layerIdToPtr;
		mapLayerIdToPtr( dnn, layerIdToPtr );

		layerToParamDiffBlobsSum.DeleteAll();
		layerToGradientHistory.DeleteAll();

		int size;
		archive >> size;
		for( int i = 0; i < size; ++i ) {
			CString layerId;
			archive >> layerId;
			CDiffBlobSum& blobSum = layerToParamDiffBlobsSum.GetOrCreateValue( layerIdToPtr[layerId] );
			archive >> blobSum.Count;
			SerializeBlobs( mathEngine, archive, blobSum.Sum );
		}

		archive >> size;
		for( int i = 0; i < size; ++i ) {
			CString layerId;
			archive >> layerId;
			SerializeBlobs( mathEngine, archive, layerToGradientHistory.GetOrCreateValue( layerIdToPtr[layerId] ) );
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
}

static const int DnnSimpleGradientSolverVersion = 0;

void CDnnSimpleGradientSolver::Serialize( CArchive& archive, CDnn& dnn )
{
	archive.SerializeVersion( DnnSimpleGradientSolverVersion );
	CDnnSolver::Serialize( archive, dnn );
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
}

// Turns on the AMSGrad mode; you can call this method only before training
void CDnnAdaptiveGradientSolver::EnableAmsGrad( bool enable )
{
	Reset();
	isAmsGradEnabled = enable;
}

static const int DnnAdaptiveGradientSolver = 0;

void CDnnAdaptiveGradientSolver::Serialize( CArchive& archive, CDnn& dnn )
{
	archive.SerializeVersion( DnnAdaptiveGradientSolver );
	CDnnSolver::Serialize( archive, dnn );
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
}

// Turns on the AMSGrad mode. The solver will be reset to initial state
void CDnnNesterovGradientSolver::EnableAmsGrad( bool enable )
{
	Reset();
	isAmsGradEnabled = enable;
}

static const int DnnNesterovGradientSolverVersion = 0;

void CDnnNesterovGradientSolver::Serialize( CArchive& archive, CDnn& dnn )
{
	archive.SerializeVersion( DnnNesterovGradientSolverVersion );
	CDnnSolver::Serialize( archive, dnn );
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

////////////////////////////////////////////////////////////////////////////////////////////////////

CDnnLambGradientSolver::CDnnLambGradientSolver( IMathEngine& mathEngine ) :
	CDnnSolver( mathEngine ),
	momentDecayRate( 0.9f ),
	secondMomentDecayRate( 0.999f ),
	epsilon( 1e-6f ),
	weightDecayClip( -1.f ),
	useTrustRatio( true ),
	useNvLamb( false ),
	tempVariables( CDnnBlob::CreateVector( mathEngine, CT_Float, TV_Count ) ),
	totalGradientNorm( 1.0f )
{
}

void CDnnLambGradientSolver::ExcludeWeightDecayLayer( const char* layerName, TExcludeLayerNameMatchType type,
	int paramIndex )
{
	CExcludedLayer excludedLayer;
	excludedLayer.LayerName = layerName;
	excludedLayer.MatchType = type;
	excludedLayer.ParamIndex = paramIndex;
	excludedLayers.Add( excludedLayer );
}

static const int DnnLambGradientSolverVersion = 0;

void CDnnLambGradientSolver::Serialize( CArchive& archive, CDnn& dnn )
{
	archive.SerializeVersion( DnnLambGradientSolverVersion );
	CDnnSolver::Serialize( archive, dnn );
	archive.Serialize( momentDecayRate );
	archive.Serialize( secondMomentDecayRate );
	archive.Serialize( epsilon );
	archive.Serialize( weightDecayClip );
	archive.Serialize( useTrustRatio );
	archive.Serialize( useNvLamb );
	archive.Serialize( layersGradientNormSquare );

	int excludedLayersCount = excludedLayers.Size();
	archive.Serialize( excludedLayersCount );

	if( archive.IsLoading() ) {
		excludedLayers.SetSize( excludedLayersCount );
	}

	for( int i = 0; i < excludedLayers.Size(); ++i ) {
		archive.Serialize( excludedLayers[i].LayerName );
		int matchType = static_cast<int>( excludedLayers[i].MatchType );
		archive.Serialize( matchType );
		excludedLayers[i].MatchType = static_cast<TExcludeLayerNameMatchType>( matchType );
		archive.Serialize( excludedLayers[i].ParamIndex );
	}
}

void CDnnLambGradientSolver::TrainLayer( const CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramBlobs,
	const CObjectArray<CDnnBlob>& paramDiffBlobs, CObjectArray<CDnnBlob>& gradientHistory )
{
	if( gradientHistory.IsEmpty() ) {
		for( int j = 0; j < 2; j++ ) {
			for( int i = 0; i < paramDiffBlobs.Size(); ++i ) {
				CDnnBlob* blob = paramDiffBlobs[i]->GetClone();
				blob->Clear();
				gradientHistory.Add( blob );
			}
		}
	}

	const float rate = layer->GetBaseLearningRate() * GetLearningRate();
	const float layerWeighDecay = GetL2Regularization() * layer->GetBaseL2RegularizationMult();
	const float clipMultiplier = 1.0f / max( 1.0f, totalGradientNorm );

	CFastArray<float, TV_Count> varValues;
	varValues.SetSize( TV_Count );

	varValues[TV_MomentDecayRateVar] = momentDecayRate;
	varValues[TV_SecondMomentDecayRateVar] = secondMomentDecayRate;
	varValues[TV_OpMomentDecayRateVar] = 1.f - momentDecayRate;
	varValues[TV_OpSecondMomentDecayRateVar] = 1.f - secondMomentDecayRate;
	varValues[TV_RateVar] = -rate;
	varValues[TV_EpsilonVar] = epsilon;
	varValues[TV_WeightDecayVar] = layerWeighDecay;
	varValues[TV_ClipMultiplierVar] = clipMultiplier;
	varValues[TV_LayerNormVar] = 0.f;
	varValues[TV_TrustRatioVar] = 0.f;
	varValues[TV_L2NormVar] = 0.f;

	MathEngine().DataExchangeTyped( tempVariables->GetData(), varValues.GetPtr(), TV_Count );

	// Getting parameters affected by weight decay
	CHashTable<int> weightDecayParamIndexes;
	getWeightDecayIndices( *layer, paramBlobs.Size(), weightDecayParamIndexes );

	for( int i = 0; i < paramBlobs.Size(); ++i ) {
		int dataSize = paramBlobs[i]->GetDataSize();
		CDnnBlob* moment = gradientHistory[i];
		CDnnBlob* secondMoment = gradientHistory[i + paramDiffBlobs.Size() * GHT_SecondMomentAverage];

		if( tempBlob == 0 || tempBlob->GetDataSize() != paramDiffBlobs[i]->GetDataSize() ) {
			tempBlob = CDnnBlob::CreateVector( MathEngine(), CT_Float, paramDiffBlobs[i]->GetDataSize() );
		}

		CPtr<CDnnBlob> paramDiffBlob = paramDiffBlobs[i];

		if( useNvLamb ) {
			MathEngine().VectorMultiply( paramDiffBlob->GetData(), paramDiffBlob->GetData(), dataSize,
				tempVariables->GetData( { TV_ClipMultiplierVar } ) );
		}

		// Update the historical gradient
		MathEngine().VectorMultiply( moment->GetData(), moment->GetData(), dataSize, 
			tempVariables->GetData( { TV_MomentDecayRateVar } ) );
		MathEngine().VectorMultiplyAndAdd( moment->GetData(), paramDiffBlob->GetData(),
			moment->GetData(), dataSize, tempVariables->GetData( { TV_OpMomentDecayRateVar } ) );

		// Calculate the historical average squared gradient
		MathEngine().VectorEltwiseMultiply( paramDiffBlob->GetData(), paramDiffBlob->GetData(),
			tempBlob->GetData(), dataSize );

		// Add squared L2-norm for calculation of L2-norm of the whole mode
		if( useNvLamb ) {
			const float invSquareClipMultiplier = 1.0f / ( clipMultiplier * clipMultiplier );
			MathEngine().VectorSum( tempBlob->GetData(), dataSize, tempVariables->GetData( { TV_LayerNormVar } ) );
			layersGradientNormSquare.Add( invSquareClipMultiplier * tempVariables->GetData( { TV_LayerNormVar } ).GetValue() );
		}

		MathEngine().VectorMultiply( secondMoment->GetData(), secondMoment->GetData(), dataSize,
			tempVariables->GetData( { TV_SecondMomentDecayRateVar } ) );
		MathEngine().VectorMultiplyAndAdd( secondMoment->GetData(), tempBlob->GetData(),
			secondMoment->GetData(), dataSize, tempVariables->GetData( { TV_OpSecondMomentDecayRateVar } ) );

		// square root of the second moment
		MathEngine().VectorSqrt( secondMoment->GetData(), tempBlob->GetData(), dataSize );

		// add epsilon before division
		MathEngine().VectorAddValue( tempBlob->GetData(), tempBlob->GetData(), dataSize,
			tempVariables->GetData( { TV_EpsilonVar } ));

		// divide historical gradient by the square root
		MathEngine().VectorEltwiseDivide( moment->GetData(), tempBlob->GetData(),
			tempBlob->GetData(), dataSize );

		// weightDecay
		if( weightDecayParamIndexes.Has( i ) && layerWeighDecay > 0 ) {
			MathEngine().VectorMultiplyAndAdd( tempBlob->GetData(), paramBlobs[i]->GetData(),
				tempBlob->GetData(), tempBlob->GetDataSize(), tempVariables->GetData( { TV_WeightDecayVar } ) );
		}

		if( useTrustRatio ) {
			// apply normalizing multiplier
			calcNormalizeMultiplier( *paramBlobs[i], *tempBlob, tempVariables->GetData( { TV_TrustRatioVar } ) );
			MathEngine().VectorMultiply( tempBlob->GetData(),
				tempBlob->GetData(), dataSize, tempVariables->GetData( { TV_TrustRatioVar } ) );
		}

		// adding gradient
		MathEngine().VectorMultiplyAndAdd( paramBlobs[i]->GetData(), tempBlob->GetData(),
			paramBlobs[i]->GetData(), dataSize, tempVariables->GetData( { TV_RateVar } ) );
	}
}

// L2 norm of a vector
float CDnnLambGradientSolver::calcL2Norm( const CConstFloatHandle& data, int dataSize ) const
{
	tempVariables->GetData( { TV_L2NormVar } ).SetValue( 0.f );
	MathEngine().VectorDotProduct( data, data, dataSize, tempVariables->GetData( { TV_L2NormVar } ) );

	const float result = tempVariables->GetData( { TV_L2NormVar } ).GetValue();
	return static_cast<float>( sqrt( result ) );
}

// Parameter indices, used in weightDecay
void CDnnLambGradientSolver::getWeightDecayIndices( const CBaseLayer& layer, int paramsCount,
	CHashTable<int>& indexes ) const
{
	CHashTable<int> excludedIndexes;
	const CString layerName = layer.GetName();
	for( int i = 0; i < excludedLayers.Size(); i++ ) {
		const CExcludedLayer& excludedLayer = excludedLayers[i];
		switch( excludedLayer.MatchType ) {
			case ELNMT_Exact:
				if( excludedLayer.LayerName == layerName ) {
					excludedIndexes.Add( excludedLayer.ParamIndex );
				}
				break;
			case ELNMT_Include:
				if( layerName.Find( excludedLayer.LayerName ) != NotFound ) {
					excludedIndexes.Add( excludedLayer.ParamIndex );
				}
				break;
			default:
				break;

		}
	}

	if( excludedIndexes.Has( -1 ) ) {
		return;
	}

	for( int i = 0; i < paramsCount; i++ ) {
		if( !excludedIndexes.Has( i ) ) {
			indexes.Add( i );
		}
	}
}

// Calculate normalizing multiplier
void CDnnLambGradientSolver::calcNormalizeMultiplier( const CDnnBlob& weights, const CDnnBlob& update,
	const CFloatHandle& multiplierVar ) const
{
	float weightNorm = calcL2Norm( weights.GetData(), weights.GetDataSize() );
	if( weightDecayClip > 0 ) {
		weightNorm = min( weightNorm, weightDecayClip );
	}

	const float updateNorm = calcL2Norm( update.GetData(), update.GetDataSize() );

	float multiplier = 0;
	if( weightNorm > 0 && updateNorm > 0 ) {
		multiplier = weightNorm / updateNorm;
	} else {
		multiplier = 1.0f;
	}
	multiplierVar.SetValue( multiplier );
}

void CDnnLambGradientSolver::OnTrain()
{
	if( !useNvLamb ) {
		return;
	}

	if( layersGradientNormSquare.IsEmpty() ) {
		totalGradientNorm = 1.0f;
	} else {
		totalGradientNorm = 0;
		for( int i = 0; i < layersGradientNormSquare.Size(); ++i ) {
			totalGradientNorm += layersGradientNormSquare[i];
		}
		totalGradientNorm = sqrtf( totalGradientNorm );
	}

	// Preventing division by zero
	if( totalGradientNorm < epsilon ) {
		totalGradientNorm = 1.0f;
	}

	layersGradientNormSquare.Empty();
}

} // namespace NeoML
