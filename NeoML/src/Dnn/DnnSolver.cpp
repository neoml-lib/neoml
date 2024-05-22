/* Copyright © 2017-2024 ABBYY

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

#include <cmath>
#include <cfloat>

#include <NeoML/Dnn/DnnSolver.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

// For LAMB solver init
#include <NeoML/Dnn/Layers/BatchNormalizationLayer.h>
#include <NeoML/Dnn/Layers/ObjectNormalizationLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/3dConvLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/TimeConvLayer.h>
#include <NeoML/Dnn/Layers/TransposedConvLayer.h>

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

//---------------------------------------------------------------------------------------------------------------------

namespace {
REGISTER_NEOML_SOLVER( CDnnSimpleGradientSolver, "NeoMLDnnSimpleGradientSolver" )
REGISTER_NEOML_SOLVER( CDnnAdaptiveGradientSolver, "NeoMLDnnAdaptiveGradientSolver" )
REGISTER_NEOML_SOLVER( CDnnNesterovGradientSolver, "NeoMLDnnNesterovGradientSolver" )
REGISTER_NEOML_SOLVER( CDnnLambGradientSolver, "NeoMLDnnLambGradientSolver" )
}

//---------------------------------------------------------------------------------------------------------------------

constexpr const char* const layerPathSeparator = "/";

CDnnSolver::CDnnSolver( IMathEngine& _mathEngine ) :
	mathEngine( _mathEngine ),
	gradParams( CDnnBlob::CreateVector( mathEngine, CT_Float, 2 ) ),
	learningRate( 0.01f ),
	regularizationL2( 0.f ),
	regularizationL1( 0.f ),
	maxGradientNorm( -1.f ),
	clipGradientMin( -FLT_MAX ),
	clipGradientMax( FLT_MAX )
{
}

CFloatHandle CDnnSolver::TempData()
{
	return temporaryBlob->GetData();
}

bool CDnnSolver::ReInitTempBlob( int dataSize )
{
	if( temporaryBlob == nullptr || temporaryBlob->GetDataSize() < dataSize ) {
		temporaryBlob = CDnnBlob::CreateVector( MathEngine(), CT_Float, dataSize );
		return true;
	}
	return false;
}

// Calculates the layer parameter gradients to then use them in Train method
void CDnnSolver::AddDiff( const CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramDiffBlobs, bool sharedWeights )
{
	NeoAssert( layer != nullptr );
	if( MathEngine().IsDistributed() && !layersToReduce.Has( layer ) ) {
		layersToReduce.Add( layer );
		reduceOrder.Add( layer );
	}

	CDiffBlobSum& paramDiffBlobsSum = layerToParamDiffBlobsSum.GetOrCreateValue( layer->GetPath( layerPathSeparator ) );
	if( paramDiffBlobsSum.LayerOwner == nullptr ) {
		paramDiffBlobsSum.LayerOwner = layer;
	}
	if( !sharedWeights ) {
		++paramDiffBlobsSum.Count;
	}

	if( paramDiffBlobsSum.Sum.IsEmpty() ) {
		paramDiffBlobs.CopyTo( paramDiffBlobsSum.Sum );
	} else {
		NeoAssert( paramDiffBlobsSum.Sum.Size() == paramDiffBlobs.Size() );
		for( int i = 0; i < paramDiffBlobs.Size(); i++ ) {
			paramDiffBlobsSum.Sum[i]->Add( paramDiffBlobs[i] );
		}
	}
}

// Modifies the trainable parameters of the network layers, using the accumulated gradient values 
// and the history of previous modifications (moment, etc.)
void CDnnSolver::Train( float distributedCoeff )
{
	OnTrain();

	for( TMapPosition pos = layerToParamDiffBlobsSum.GetFirstPosition(); pos != NotFound;
		pos = layerToParamDiffBlobsSum.GetNextPosition( pos ) )
	{
		const CString layerPath = layerToParamDiffBlobsSum.GetKey( pos );
		CDiffBlobSum& paramDiffBlobsSum = layerToParamDiffBlobsSum.GetValue( pos );
		if( paramDiffBlobsSum.Sum.IsEmpty() ) {
			continue;
		}
		const CBaseLayer* layer = paramDiffBlobsSum.LayerOwner;
		NeoAssert( layer != nullptr );
		NeoAssert( paramDiffBlobsSum.Count > 0 );

		// Take the average of the gradients to simulate that the elements from all runs were in the same batch
		// TODO: weighted average
		if( paramDiffBlobsSum.Count > 1 ) {
			const float oneDivEpoch( 1.f / paramDiffBlobsSum.Count );
			for( int i = 0; i < paramDiffBlobsSum.Sum.Size(); i++ ) {
				MathEngine().VectorMultiply( paramDiffBlobsSum.Sum[i]->GetData(), paramDiffBlobsSum.Sum[i]->GetData(),
					paramDiffBlobsSum.Sum[i]->GetDataSize(), oneDivEpoch );
			}
		}

		clipGradients( paramDiffBlobsSum.Sum );

		// Train the layer based on the calculated diff data
		TrainLayer( layer, layer->paramBlobs, paramDiffBlobsSum.Sum, layerToGradientHistory.GetOrCreateValue( layerPath ) );

		// Clear the diff data
		paramDiffBlobsSum.Sum.Empty();
		paramDiffBlobsSum.Count = 0;
	}

	if( MathEngine().IsDistributed() ) {
		allReduce( distributedCoeff );
	}
}

void CDnnSolver::Reset()
{
	layerToParamDiffBlobsSum.DeleteAll();
	layerToGradientHistory.DeleteAll();
	OnReset();
}

void CDnnSolver::allReduce( float distributedCoeff )
{
	const bool isCoeffNontrivial = ::fabsf( distributedCoeff - 1.f ) >= FLT_EPSILON;
	for( int i = 0; i < reduceOrder.Size(); ++i ) {
		if( !reduceOrder[i]->IsLearnable() || !reduceOrder[i]->IsLearningEnabled() ) {
			continue;
		}
		const CObjectArray<CDnnBlob>& params = reduceOrder[i]->paramBlobs;
		for( int j = 0; j < params.Size(); j++ ) {
			if( isCoeffNontrivial ) {
				MathEngine().VectorMultiply( params[j]->GetData(), params[j]->GetData(),
					params[j]->GetDataSize(), distributedCoeff );
			}
			MathEngine().AllReduce( params[j]->GetData(), params[j]->GetDataSize() );
		}
	}
}

void CDnnSolver::clip( const CObjectArray<CDnnBlob>& paramDiffBlobs )
{
	if( clipGradientMin <= -FLT_MAX && clipGradientMax >= FLT_MAX ) {
		return;
	}

	for( int i = 0; i < paramDiffBlobs.Size(); ++i ) {
		MathEngine().VectorMinMax( paramDiffBlobs[i]->GetData(), paramDiffBlobs[i]->GetData(),
			paramDiffBlobs[i]->GetDataSize(), clipGradientMin, clipGradientMax );
	}
}

void CDnnSolver::clipGradients( const CObjectArray<CDnnBlob>& paramDiffBlobs )
{
	if( paramDiffBlobs.Size() == 0 ) {
		return;
	}

	clip( paramDiffBlobs );

	if( maxGradientNorm < 0 ) {
		return;
	}

	// Calculate the parameter gradient norm
	CFloatHandle tempVar = gradParams->GetData( { 0 } );
	CFloatHandle gradVar = gradParams->GetData( { 1 } );

	MathEngine().VectorDotProduct( paramDiffBlobs[0]->GetData(), paramDiffBlobs[0]->GetData(),
		paramDiffBlobs[0]->GetDataSize(), gradVar );
	for( int i = 1; i < paramDiffBlobs.Size(); ++i ) {
		MathEngine().VectorDotProduct( paramDiffBlobs[i]->GetData(), paramDiffBlobs[i]->GetData(),
			paramDiffBlobs[i]->GetDataSize(), tempVar );
		MathEngine().VectorAdd( gradVar, tempVar, gradVar, 1 );
	}
	NeoPresume( std::isfinite( gradVar.GetValue() ) );
	MathEngine().VectorSqrt( gradVar, gradVar, 1 );

	// Calculate scale
	MathEngine().VectorMax( gradVar, maxGradientNorm, gradVar, 1 );
	MathEngine().VectorInv( gradVar, tempVar, 1 );
	MathEngine().VectorMultiply( tempVar, tempVar, 1, maxGradientNorm );

	// Decrease the gradient
	for( int i = 0; i < paramDiffBlobs.Size(); ++i ) {
		MathEngine().VectorMultiply( paramDiffBlobs[i]->GetData(), paramDiffBlobs[i]->GetData(),
			paramDiffBlobs[i]->GetDataSize(), tempVar );
	}
}

constexpr int DnnSolverVersion = 2;

void CDnnSolver::Serialize( CArchive& archive, const CDnn& dnn )
{
	const int version = archive.SerializeVersion( DnnSolverVersion );
	int size = layerToParamDiffBlobsSum.Size();
	archive.Serialize( size );

	if( archive.IsStoring() ) {
		CArray<CString> path;
		for( auto& blobsSum : layerToParamDiffBlobsSum ) {
			serializePath( archive, path, blobsSum.Value.LayerOwner );
			serializeDiffBlobSum( archive, blobsSum.Value, /*layer*/nullptr );

			serializeGradientHistory( archive, blobsSum.Key );
		}
	} else if( archive.IsLoading() ) {
		layerToParamDiffBlobsSum.DeleteAll();
		layerToGradientHistory.DeleteAll();
		layersToReduce.DeleteAll();
		reduceOrder.DeleteAll();

		if( version >= 2 ) {
			CArray<CString> path;
			for( int i = 0; i < size; ++i ) {
				serializePath( archive, path, /*layer*/nullptr );
				const CString layerPath = JoinStrings( path, layerPathSeparator );
				serializeDiffBlobSum( archive, layerToParamDiffBlobsSum.CreateValue( layerPath ), dnn.GetLayer( path ) );

				serializeGradientHistory( archive, layerPath );
			}
		} else {
			serializeLoadMapsPrevVersion( archive, dnn, size );
		}
	} else {
		NeoAssert( false );
	}

	archive.Serialize( learningRate );
	archive.Serialize( regularizationL1 );
	archive.Serialize( regularizationL2 );
	archive.Serialize( maxGradientNorm );
	if( version >= 1 ) {
		archive.Serialize( clipGradientMin );
		archive.Serialize( clipGradientMax );
	} else {
		clipGradientMin = -FLT_MAX;
		clipGradientMax = FLT_MAX;
	}
}

void CDnnSolver::serializeLoadMapsPrevVersion( CArchive& archive, const CDnn& dnn, int size )
{
	CMap<CString, CArray<CString>> layerPrevIdToPath;
	auto mapLayerIdToPath = [&]( const CDnnLayerGraph& dnn, auto& mapLayerIdToPath ) -> void
	{
		CArray<const char*> layerNames;
		dnn.GetLayerList( layerNames );
		for( const char* layerName : layerNames ) {
			const CBaseLayer* layer = dnn.GetLayer( layerName );
			const CString layerPath = layer->GetPath( "" );
			CArray<CString>& path = layerPrevIdToPath.GetOrCreateValue( layerPath );
			layer->GetPath( path );
			NeoAssert( path.Size() > 0 );
			const CCompositeLayer* composite = dynamic_cast<const CCompositeLayer*>( layer );
			if( composite != nullptr ) {
				mapLayerIdToPath( *composite, mapLayerIdToPath );
			}
		}
	};
	mapLayerIdToPath( dnn, mapLayerIdToPath );

	auto convertOldIdToLayerPath = [&]( const CBaseLayer** layer )
	{
		CString layerId;
		archive >> layerId;
		const CArray<CString>& path = layerPrevIdToPath[layerId];
		if( layer != nullptr ) {
			*layer = dnn.GetLayer( path );
		}
		return JoinStrings( path, layerPathSeparator );
	};

	for( int i = 0; i < size; ++i ) {
		const CBaseLayer* layer = nullptr;
		const CString layerPath = convertOldIdToLayerPath( &layer );
		serializeDiffBlobSum( archive, layerToParamDiffBlobsSum.CreateValue( layerPath ), layer );
	}

	archive >> size;
	for( int i = 0; i < size; ++i ) {
		const CString layerPath = convertOldIdToLayerPath( nullptr );
		SerializeBlobs( mathEngine, archive, layerToGradientHistory.CreateValue( layerPath ) );
	}
}

void CDnnSolver::serializePath( CArchive& archive, CArray<CString>& path, const CBaseLayer* layer )
{
	if( archive.IsStoring() ) {
		NeoAssert( layer != nullptr );
		layer->GetPath( path );
	}
	archive.Serialize( path );
	NeoAssert( path.Size() > 0 );
}

void CDnnSolver::serializeDiffBlobSum( CArchive& archive, CDiffBlobSum& blobsSum, const CBaseLayer* layer )
{
	archive.Serialize( blobsSum.Count );
	SerializeBlobs( mathEngine, archive, blobsSum.Sum );
	if( archive.IsLoading() ) {
		blobsSum.LayerOwner = layer;
	}
}

void CDnnSolver::serializeGradientHistory( CArchive& archive, const CString& layerPath )
{
	if( archive.IsStoring() ) {
		const bool hasGradientHistory = layerToGradientHistory.Has( layerPath );
		archive << hasGradientHistory;
		if( hasGradientHistory ) {
			SerializeBlobs( mathEngine, archive, layerToGradientHistory.GetOrCreateValue( layerPath ) );
		}
	} else if( archive.IsLoading() ) {
		bool hasGradientHistory;
		archive >> hasGradientHistory;
		if( hasGradientHistory ) {
			SerializeBlobs( mathEngine, archive, layerToGradientHistory.CreateValue( layerPath ) );
		}
	} else {
		NeoAssert( false );
	}
}

//---------------------------------------------------------------------------------------------------------------------

CDnnSimpleGradientSolver::CDnnSimpleGradientSolver( IMathEngine& mathEngine ) :
	CDnnSolver( mathEngine ),
	momentDecayRate( 0.9f ),
	isInCompatibilityMode( false )
{
}

constexpr int DnnSimpleGradientSolverVersion = 0;

void CDnnSimpleGradientSolver::Serialize( CArchive& archive, const CDnn& dnn )
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

	// Add regularization and add diffs to parameters
	float rate = layer->GetLearningRate() * GetLearningRate();
	const float regL1 = layer->GetL1RegularizationMult() * GetL1Regularization();
	const float regL2 = layer->GetL2RegularizationMult() * GetL2Regularization();
	// Set the values of the variables
	const float opMomentDecayRate = 1 - momentDecayRate;
	const float opRegL2MomentDecayRate = isInCompatibilityMode ? ( ( 1 - momentDecayRate ) * regL2 ) : ( -rate * regL2 );
	const float multL1 = isInCompatibilityMode ? 1.f : -rate;
	rate = ( -rate );

	for(int i = 0; i < paramBlobs.Size(); ++i) {
		const int dataSize = paramBlobs[i]->GetDataSize();
		// Update the gradient in history
		MathEngine().VectorMultiply( gradientHistory[i]->GetData(),
			gradientHistory[i]->GetData(), dataSize, momentDecayRate );
		MathEngine().VectorMultiplyAndAdd( gradientHistory[i]->GetData(), paramDiffBlobs[i]->GetData(), 
			gradientHistory[i]->GetData(), dataSize, isInCompatibilityMode ? opMomentDecayRate : rate );

		if(regL2 > 0) {
			MathEngine().VectorMultiplyAndAdd( gradientHistory[i]->GetData(), paramBlobs[i]->GetData(),
				gradientHistory[i]->GetData(), dataSize, opRegL2MomentDecayRate );
		}
		if(regL1 > 0) {
			MathEngine().VectorL1DiffAdd( gradientHistory[i]->GetData(), paramBlobs[i]->GetData(),
				gradientHistory[i]->GetData(), dataSize, regL1, multL1 );
		}

		// Add regularization and gradient
		if( isInCompatibilityMode ) {
			MathEngine().VectorMultiplyAndAdd( paramBlobs[i]->GetData(), gradientHistory[i]->GetData(),
				paramBlobs[i]->GetData(), dataSize, rate );
		} else {
			MathEngine().VectorAdd( paramBlobs[i]->GetData(), gradientHistory[i]->GetData(),
				paramBlobs[i]->GetData(), dataSize );
		}
	}
}

//---------------------------------------------------------------------------------------------------------------------

CDnnAdaptiveGradientSolver::CDnnAdaptiveGradientSolver( IMathEngine& mathEngine ) :
	CDnnSolver( mathEngine ),
	momentDecayRate(0.9f),
	momentDecayRateN(1.f),
	secondMomentDecayRate(0.99f),
	secondMomentDecayRateN(1.f),
	epsilon(1e-6f),
	isAmsGradEnabled( false ),
	isDecoupledWeightDecay( false ),
	isInCompatibilityMode( false )
{
}

// Turns on the AMSGrad mode; you can call this method only before training
void CDnnAdaptiveGradientSolver::EnableAmsGrad( bool enable )
{
	Reset();
	isAmsGradEnabled = enable;
}

void CDnnAdaptiveGradientSolver::EnableDecoupledWeightDecay( bool enable )
{
	Reset();
	isDecoupledWeightDecay = enable;
}

constexpr int DnnAdaptiveGradientSolver = 1;

void CDnnAdaptiveGradientSolver::Serialize( CArchive& archive, const CDnn& dnn )
{
	const int version = archive.SerializeVersion( DnnAdaptiveGradientSolver );
	CDnnSolver::Serialize( archive, dnn );
	archive.Serialize( momentDecayRate );
	archive.Serialize( momentDecayRateN );
	archive.Serialize( secondMomentDecayRate );
	archive.Serialize( secondMomentDecayRateN );
	archive.Serialize( epsilon );
	archive.Serialize( isAmsGradEnabled );
	if( version < 1 ) {
		isDecoupledWeightDecay = false;
	} else {
		archive.Serialize( isDecoupledWeightDecay );
	}
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

// Add regularization
static CConstFloatHandle addRegularization( IMathEngine& mathEngine, CConstFloatHandle diffBlob, CDnnBlob* params,
	float regL1, float regL2, float multL1, CFloatHandle tempBlob )
{
	if( regL2 > 0 ) {
		mathEngine.VectorMultiplyAndAdd( diffBlob, params->GetData(), tempBlob, params->GetDataSize(), regL2 );
		diffBlob = tempBlob;
	}
	if( regL1 > 0 ) {
		mathEngine.VectorL1DiffAdd( diffBlob, params->GetData(), tempBlob, params->GetDataSize(), regL1, multL1 );
		diffBlob = tempBlob;
	}
	return diffBlob;
}

void CDnnAdaptiveGradientSolver::TrainLayer( const CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramBlobs,
	const CObjectArray<CDnnBlob>& paramDiffBlobs, CObjectArray<CDnnBlob>& gradientHistory )
{
	if(gradientHistory.Size() == 0) {
		// Create blobs
		const int gradientHistoryTypeCount = IsAmsGradEnabled() ? GHTC_AmsGrad : GHTC_Default;
		for( int j = 0; j < gradientHistoryTypeCount; j++ ) {
			for(int i = 0; i < paramDiffBlobs.Size(); ++i) {
				CDnnBlob* blob = paramDiffBlobs[i]->GetClone();
				blob->Clear();
				gradientHistory.Add( blob );
			}
		}
	}

	// Add regularization and add diffs to parameters
	float rate = layer->GetLearningRate() * GetLearningRate() * sqrtf( 1 - secondMomentDecayRateN );
	if( !isInCompatibilityMode ) {
		rate /= ( 1 - momentDecayRateN );
	}
	const float regL1 = layer->GetL1RegularizationMult() * GetL1Regularization();
	const float regL2 = layer->GetL2RegularizationMult() * GetL2Regularization();
	// Set the values of the variables
	const float opMomentDecayRate = 1 - momentDecayRate;
	const float opSecondMomentDecayRate = 1 - secondMomentDecayRate;
	const float multL1 = 1.f;
	rate = ( -rate );

	for( int i = 0; i < paramBlobs.Size(); ++i ) {
		const int dataSize = paramBlobs[i]->GetDataSize();
		CDnnBlob* moment = gradientHistory[i];
		CDnnBlob* secondMoment = gradientHistory[i + paramDiffBlobs.Size() * GHT_SecondMomentAverage];
		const CDnnBlob* paramDiffBlob = paramDiffBlobs[i];

		ReInitTempBlob( paramDiffBlob->GetDataSize() );

		CConstFloatHandle paramDiffHandle = paramDiffBlob->GetData();
		if( !IsDecoupledWeightDecay() ) {
			paramDiffHandle = addRegularization( MathEngine(), paramDiffHandle, paramBlobs[i], regL1, regL2,
				multL1, TempData() );
		}

		// Update the historical gradient
		MathEngine().VectorMultiply( moment->GetData(), moment->GetData(), dataSize, momentDecayRate );
		MathEngine().VectorMultiplyAndAdd( moment->GetData(), paramDiffHandle,
			moment->GetData(), dataSize, opMomentDecayRate );
		// Calculate the historical average squared gradient
		MathEngine().VectorEltwiseMultiply( paramDiffHandle, paramDiffHandle, TempData(), dataSize );
		MathEngine().VectorMultiply( secondMoment->GetData(), secondMoment->GetData(), dataSize,
			secondMomentDecayRate );
		MathEngine().VectorMultiplyAndAdd( secondMoment->GetData(), TempData(),
			secondMoment->GetData(), dataSize, opSecondMomentDecayRate );
		if( IsAmsGradEnabled() ) {
			// Update the maximum of the average
			CDnnBlob* secondMomentMaxAverage = gradientHistory[i + paramDiffBlobs.Size() * GHT_SecondMomentMaxAverage];
			MathEngine().VectorEltwiseMax( secondMomentMaxAverage->GetData(), secondMoment->GetData(),
				secondMomentMaxAverage->GetData(), secondMomentMaxAverage->GetDataSize() );
			// Calculate the square root of the historical maximum of average squared gradient
			MathEngine().VectorSqrt( secondMomentMaxAverage->GetData(), TempData(), dataSize );
		} else {
			// Calculate the square root of the historical average squared gradient
			MathEngine().VectorSqrt( secondMoment->GetData(), TempData(), dataSize );
		}
		// Add epsilon before dividing
		MathEngine().VectorAddValue( TempData(), TempData(), dataSize, epsilon );
		// Divide the historical gradient by the square root
		MathEngine().VectorEltwiseDivide( moment->GetData(), TempData(), TempData(), dataSize );

		CConstFloatHandle tempHandle = TempData();
		if( IsDecoupledWeightDecay() ) {
			tempHandle = addRegularization( MathEngine(), TempData(), paramBlobs[i], regL1, regL2,
				multL1, TempData() );
		}
		// Add the gradient
		MathEngine().VectorMultiplyAndAdd( paramBlobs[i]->GetData(), tempHandle,
			paramBlobs[i]->GetData(), dataSize, rate );
	}
}

//---------------------------------------------------------------------------------------------------------------------

CDnnNesterovGradientSolver::CDnnNesterovGradientSolver( IMathEngine& mathEngine ) :
	CDnnSolver( mathEngine ),
	momentDecayRate( 0.9f ),
	secondMomentDecayRate( 0.99f ),
	secondMomentDecayRateN( 1.f ),
	epsilon( 1e-6f ),
	isAmsGradEnabled( false ),
	isDecoupledWeightDecay( false ),
	trainCount( 0 ),
	productMuT( 1.f )
{
}

// Turns on the AMSGrad mode. The solver will be reset to initial state
void CDnnNesterovGradientSolver::EnableAmsGrad( bool enable )
{
	Reset();
	isAmsGradEnabled = enable;
}

void CDnnNesterovGradientSolver::EnableDecoupledWeightDecay( bool enable )
{
	Reset();
	isDecoupledWeightDecay = enable;
}

constexpr int DnnNesterovGradientSolverVersion = 1;

void CDnnNesterovGradientSolver::Serialize( CArchive& archive, const CDnn& dnn )
{
	const int version = archive.SerializeVersion( DnnNesterovGradientSolverVersion );
	CDnnSolver::Serialize( archive, dnn );
	archive.Serialize( momentDecayRate );
	archive.Serialize( secondMomentDecayRate );
	archive.Serialize( secondMomentDecayRateN );
	archive.Serialize( epsilon );
	archive.Serialize( isAmsGradEnabled );
	if( version < 1 ) {
		isDecoupledWeightDecay = false;
	} else {
		archive.Serialize( isDecoupledWeightDecay );
	}
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
	float rate = layer->GetLearningRate() * GetLearningRate();
	const float regL1 = layer->GetL1RegularizationMult() * GetL1Regularization();
	const float regL2 = layer->GetL2RegularizationMult() * GetL2Regularization();
	// Set the values of the variables
	const float opMomentDecayRate = 1 - momentDecayRate;
	const float opSecondMomentDecayRate = 1 - secondMomentDecayRate;
	const float multL1 = 1.f;
	const float mBarGradMult = ( 1.f - muT ) / ( 1.f - productMuT );
	const float mBarMomentMult = muTPlusOne / ( 1.f - productMuT * muTPlusOne );
	const float invOpSecondMomentDecayRateN = 1 / ( 1 - secondMomentDecayRateN );
	rate = ( -rate );

	for( int i = 0; i < paramBlobs.Size(); ++i ) {
		const int dataSize = paramBlobs[i]->GetDataSize();
		CDnnBlob* moment = gradientHistory[i];
		CDnnBlob* secondMoment = gradientHistory[i + paramDiffBlobs.Size() * GHT_SecondMomentAverage];
		const CDnnBlob* paramDiffBlob = paramDiffBlobs[i];

		const int paramDiffBlobSize = paramDiffBlob->GetDataSize();
		if( ReInitTempBlob( paramDiffBlobSize ) ) {
			if( mBarBlob == nullptr || mBarBlob->GetDataSize() < paramDiffBlobSize ) {
				mBarBlob = CDnnBlob::CreateVector( MathEngine(), CT_Float, paramDiffBlobSize );
			}
			MathEngine().VectorCopy( mBarBlob->GetData(), TempData(), paramDiffBlobSize );
		}

		CConstFloatHandle paramDiffHandle = paramDiffBlob->GetData();
		if( !IsDecoupledWeightDecay() ) {
			paramDiffHandle = addRegularization( MathEngine(), paramDiffHandle, paramBlobs[i], regL1, regL2,
				multL1, TempData() );
		}

		// Update the historical gradient
		MathEngine().VectorMultiply( moment->GetData(), moment->GetData(), dataSize, momentDecayRate );
		MathEngine().VectorMultiplyAndAdd( moment->GetData(), paramDiffHandle,
			moment->GetData(), dataSize, opMomentDecayRate );
		// Calculate the historical average squared gradient
		MathEngine().VectorEltwiseMultiply( paramDiffHandle, paramDiffHandle, TempData(), dataSize );
		MathEngine().VectorMultiply( secondMoment->GetData(), secondMoment->GetData(), dataSize,
			secondMomentDecayRate );
		MathEngine().VectorMultiplyAndAdd( secondMoment->GetData(), TempData(),
			secondMoment->GetData(), dataSize, opSecondMomentDecayRate );

		// Calculate the auxiliary variables (notations taken from the reference paper)
		// m with a dash
		CFloatHandle mBar = mBarBlob->GetData();
		MathEngine().VectorMultiply( paramDiffHandle, mBar, dataSize, mBarGradMult );
		MathEngine().VectorMultiplyAndAdd( mBar, moment->GetData(), mBar, dataSize, mBarMomentMult );

		// sqrt(n with a hat) + eps
		if( IsAmsGradEnabled() ) {
			// Update the maximum average
			CDnnBlob* secondMomentMaxAverage = gradientHistory[i + paramDiffBlobs.Size() * GHT_SecondMomentMaxAverage];
			MathEngine().VectorEltwiseMax( secondMomentMaxAverage->GetData(), secondMoment->GetData(),
				secondMomentMaxAverage->GetData(), secondMomentMaxAverage->GetDataSize() );
			// n with a hat calculated for the maximum of the second moment moving mean
			MathEngine().VectorMultiply( secondMomentMaxAverage->GetData(), TempData(), dataSize,
				invOpSecondMomentDecayRateN );
		} else {
			// n with a hat calculated for the second momentum moving mean
			MathEngine().VectorMultiply( secondMoment->GetData(), TempData(), dataSize,
				invOpSecondMomentDecayRateN );
		}
		MathEngine().VectorSqrt( TempData(), TempData(), dataSize );
		MathEngine().VectorAddValue( TempData(), TempData(), dataSize, epsilon );
		// Calculate the final diff
		MathEngine().VectorEltwiseDivide( mBar, TempData(), TempData(), dataSize );

		CConstFloatHandle tempHandle = TempData();
		if( IsDecoupledWeightDecay() ) {
			tempHandle = addRegularization( MathEngine(), TempData(), paramBlobs[i], regL1, regL2,
				multL1, TempData() );
		}
		// Update parameters
		MathEngine().VectorMultiplyAndAdd( paramBlobs[i]->GetData(), tempHandle,
			paramBlobs[i]->GetData(), dataSize, rate );
	}
}

//---------------------------------------------------------------------------------------------------------------------

CDnnLambGradientSolver::CDnnLambGradientSolver( IMathEngine& mathEngine ) :
	CDnnSolver( mathEngine ),
	momentDecayRate( 0.9f ),
	secondMomentDecayRate( 0.999f ),
	epsilon( 1e-6f ),
	weightDecayClip( -1.f ),
	useTrustRatio( true ),
	useNvLamb( false ),
	normL2Var( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	totalGradientNorm( 1.0f )
{
	SetL2Regularization( 0.01f );
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

void CDnnLambGradientSolver::ExcludeBiasParamLayers()
{
	ExcludeWeightDecayLayer<CBatchNormalizationLayer>( 0 );
	ExcludeWeightDecayLayer<CObjectNormalizationLayer>( -1 );
	ExcludeWeightDecayLayer<CFullyConnectedLayer>( 1 );
	ExcludeWeightDecayLayer<CTimeConvLayer>( 1 );
	ExcludeWeightDecayLayer<C3dConvLayer>( 1 );
	ExcludeWeightDecayLayer<CChannelwiseConvLayer>( 1 );
	ExcludeWeightDecayLayer<CConvLayer>( 1 );
	ExcludeWeightDecayLayer<CRleConvLayer>( 1 );
	ExcludeWeightDecayLayer<CTransposedConvLayer>( 1 );
}

constexpr int DnnLambGradientSolverVersion = 0;

void CDnnLambGradientSolver::Serialize( CArchive& archive, const CDnn& dnn )
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

	// Apply regularization and add diffs to the parameters
	float rate = layer->GetLearningRate() * GetLearningRate();
	const float layerWeighDecay = GetL2Regularization() * layer->GetL2RegularizationMult();
	const float clipMultiplier = 1.0f / max( 1.0f, totalGradientNorm );
	// Set the values of the variables
	const float opMomentDecayRate = 1.f - momentDecayRate;
	const float opSecondMomentDecayRate = 1.f - secondMomentDecayRate;
	rate = ( -rate );

	// Getting parameters affected by weight decay
	CHashTable<int> weightDecayParamIndexes;
	getWeightDecayIndices( *layer, paramBlobs.Size(), weightDecayParamIndexes );

	for( int i = 0; i < paramBlobs.Size(); ++i ) {
		const int dataSize = paramBlobs[i]->GetDataSize();
		CDnnBlob* moment = gradientHistory[i];
		CDnnBlob* secondMoment = gradientHistory[i + paramDiffBlobs.Size() * GHT_SecondMomentAverage];
		CDnnBlob* paramDiffBlob = paramDiffBlobs[i];

		const int paramDiffSize = paramDiffBlob->GetDataSize();
		ReInitTempBlob( paramDiffSize );

		CFloatHandle paramDiffHandle = paramDiffBlob->GetData();
		if( useNvLamb ) {
			MathEngine().VectorMultiply( paramDiffHandle, paramDiffHandle, dataSize, clipMultiplier );
		}

		// Update the historical gradient
		MathEngine().VectorMultiply( moment->GetData(), moment->GetData(), dataSize, momentDecayRate );
		MathEngine().VectorMultiplyAndAdd( moment->GetData(), paramDiffHandle,
			moment->GetData(), dataSize, opMomentDecayRate );

		// Calculate the historical average squared gradient
		MathEngine().VectorEltwiseMultiply( paramDiffHandle, paramDiffHandle, TempData(), dataSize );

		// Add squared L2-norm for calculation of L2-norm of the whole mode
		if( useNvLamb ) {
			const float invSquareClipMultiplier = 1.0f / ( clipMultiplier * clipMultiplier );
			//normL2Var->GetData().SetValue( 0.f ); // CUDA sync
			MathEngine().VectorSum( TempData(), dataSize, normL2Var->GetData() );
			const float layerNormL2 = normL2Var->GetData().GetValue(); // CUDA sync
			layersGradientNormSquare.Add( invSquareClipMultiplier * layerNormL2 );
		}

		MathEngine().VectorMultiply( secondMoment->GetData(), secondMoment->GetData(), dataSize,
			secondMomentDecayRate );
		MathEngine().VectorMultiplyAndAdd( secondMoment->GetData(), TempData(),
			secondMoment->GetData(), dataSize, opSecondMomentDecayRate );

		// Square root of the second moment
		MathEngine().VectorSqrt( secondMoment->GetData(), TempData(), dataSize );
		// Add epsilon before division
		MathEngine().VectorAddValue( TempData(), TempData(), dataSize, epsilon );
		// Divide historical gradient by the square root
		MathEngine().VectorEltwiseDivide( moment->GetData(), TempData(), TempData(), dataSize );

		// Weight decay
		if( weightDecayParamIndexes.Has( i ) && layerWeighDecay > 0 ) {
			MathEngine().VectorMultiplyAndAdd( TempData(), paramBlobs[i]->GetData(),
				TempData(), paramDiffSize, layerWeighDecay );
		}

		if( useTrustRatio ) {
			// Apply normalizing multiplier
			const float trustRatio = calcNormalizeMultiplier( *paramBlobs[i], TempBlob() );
			MathEngine().VectorMultiply( TempData(), TempData(), dataSize, trustRatio );
		}

		// Add gradient
		MathEngine().VectorMultiplyAndAdd( paramBlobs[i]->GetData(), TempData(),
			paramBlobs[i]->GetData(), dataSize, rate );
	}
}

// L2 norm of a vector devided by vector size.
float CDnnLambGradientSolver::calcL2NormAverage( const CConstFloatHandle& data, int dataSize ) const
{
	NeoAssert( dataSize > 0 );
	const float multiplier( 1.f / dataSize );
	MathEngine().VectorMultiply( data, tempNormBlob->GetData(), dataSize, multiplier );

	//normL2Var->GetData().SetValue( 0.f ); // CUDA sync
	MathEngine().VectorDotProduct( tempNormBlob->GetData(), tempNormBlob->GetData(), dataSize, normL2Var->GetData() );
	MathEngine().VectorSqrt( normL2Var->GetData(), normL2Var->GetData(), 1 );
	return normL2Var->GetData().GetValue(); // CUDA sync
}

// Parameter indices, used in weightDecay
void CDnnLambGradientSolver::getWeightDecayIndices( const CBaseLayer& layer, int paramsCount,
	CHashTable<int>& indexes ) const
{
	CHashTable<int> excludedIndexes;
	const CString layerName = layer.GetName();
	const CString layerClassName = GetLayerClass( layer );
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
			case ELNMT_LayerClass:
				if( excludedLayer.LayerName == layerClassName ) {
					excludedIndexes.Add( excludedLayer.ParamIndex );
				}
				break;
			default:
				break;

		}
	}
	static_assert( ELNMT_ItemsCount == 3, "Not all enum item are processed" );

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
float CDnnLambGradientSolver::calcNormalizeMultiplier( const CDnnBlob& weights, const CDnnBlob& update ) const
{
	const int dataSize = max( weights.GetDataSize(), update.GetDataSize() );
	if( tempNormBlob == nullptr || tempNormBlob->GetDataSize() < dataSize ) {
		tempNormBlob = CDnnBlob::CreateVector( MathEngine(), CT_Float, dataSize );
	}
	float weightNorm = calcL2NormAverage( weights.GetData(), weights.GetDataSize() );
	if( weightDecayClip > 0 ) {
		weightNorm = min( weightNorm, weightDecayClip );
	}
	const float updateNorm = calcL2NormAverage( update.GetData(), update.GetDataSize() );
	return ( weightNorm > 0 && updateNorm > 0 ) ? ( weightNorm / updateNorm ) : 1.0f;
}

void CDnnLambGradientSolver::OnTrain()
{
	if( !useNvLamb ) {
		return;
	}

	if( layersGradientNormSquare.IsEmpty() ) {
		totalGradientNorm = 1.0f;
	} else {
		// The order of numbers in layersGradientNormSquare depends on the values of layer pointers
		// As a result, cloning nets via serialization breaks it (as a result unstable failures in solver tests)
		layersGradientNormSquare.QuickSort<Ascending<float>>();
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
