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

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
REGISTER_NEOML_SOLVER( CDnnSimpleGradientSolver, "NeoMLDnnSimpleGradientSolver" )
REGISTER_NEOML_SOLVER( CDnnAdaptiveGradientSolver, "NeoMLDnnAdaptiveGradientSolver" )
REGISTER_NEOML_SOLVER( CDnnNesterovGradientSolver, "NeoMLDnnNesterovGradientSolver" )
REGISTER_NEOML_SOLVER( CDnnLambGradientSolver, "NeoMLDnnLambGradientSolver" )
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr const char* const layerPathSeparator = "/";

CDnnSolver::CDnnSolver( IMathEngine& _mathEngine, int numVariables ) :
	mathEngine( _mathEngine ),
	engineVariables( CDnnBlob::CreateVector( _mathEngine, CT_Float, numVariables ) )
{
	variables.Add( 0, numVariables );
	SetMinMaxGradientClipping( /*min*/-FLT_MAX, /*max*/FLT_MAX );
	SetLearningRate( 0.01f );
	SetL2Regularization( 0.f );
	SetL1Regularization( 0.f );
	SetMaxGradientNorm( -1.f );
}

void CDnnSolver::SetVariable( int index, float value )
{
	NeoPresume( variables.IsValidIndex( index ) );
	variables[index] = value;
	MathEngine().VectorFill( engineVariables->GetData( { index } ), value, 1 );
}

float CDnnSolver::GetVariable( int index ) const
{
	NeoPresume( variables.IsValidIndex( index ) );
	return variables[index];
}

CConstFloatHandle CDnnSolver::Var( int index ) const
{
	NeoPresume( variables.IsValidIndex( index ) );
	return engineVariables->GetData( { index } );
}

CFloatHandle CDnnSolver::UseVar( int index ) const
{
	NeoPresume( variables.IsValidIndex( index ) );
	return engineVariables->GetData( { index } );
}

// Calculates the layer parameter gradients to then use them in Train method
void CDnnSolver::AddDiff( CBaseLayer* layer, const CObjectArray<CDnnBlob>& paramDiffBlobs,
	bool sharedWeights )
{
	NeoAssert( layer != 0 );

	if( MathEngine().IsDistributed() && !layersToReduce.Has( layer ) ) {
		layersToReduce.Add( layer );
		reduceOrder.Add( layer );
	}

	CDiffBlobSum& paramDiffBlobsSum = layerToParamDiffBlobsSum.GetOrCreateValue( layer->GetPath( layerPathSeparator ) );
	if( paramDiffBlobsSum.Count == 0 ) {
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
			MathEngine().VectorFill( UseVar( TVS_OneDivEpoch ), 1.f / paramDiffBlobsSum.Count, 1 );
			for( int i = 0; i < paramDiffBlobsSum.Sum.Size(); i++ ) {
				MathEngine().VectorMultiply( paramDiffBlobsSum.Sum[i]->GetData(), paramDiffBlobsSum.Sum[i]->GetData(),
					paramDiffBlobsSum.Sum[i]->GetDataSize(), Var( TVS_OneDivEpoch ) );
			}
		}

		clipGradients( paramDiffBlobsSum.Sum );

		// Train the layer based on the calculated diff data
		TrainLayer( layer, layer->paramBlobs, paramDiffBlobsSum.Sum, layerToGradientHistory.GetOrCreateValue( layerPath ) );

		// Clear the diff data
		paramDiffBlobsSum.Sum.Empty();
		paramDiffBlobsSum.Count = 0;
	}

	if( MathEngine().IsDistributed() ){
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
	const bool isCoeffNontrivial = ( ::fabsf( distributedCoeff - 1.f ) >= FLT_EPSILON );
	if( isCoeffNontrivial ) {
		SetVariable( TVS_DistributedCoeff, distributedCoeff );
	}

	for( int i = 0; i < reduceOrder.Size(); ++i ) {
		if( !reduceOrder[i]->IsLearnable() || !reduceOrder[i]->IsLearningEnabled() ) {
			continue;
		}
		const CObjectArray<CDnnBlob>& params = reduceOrder[i]->paramBlobs;
		for( int j = 0; j < params.Size(); j++ ) {
			if( isCoeffNontrivial ) {
				MathEngine().VectorMultiply( params[j]->GetData(), params[j]->GetData(), params[j]->GetDataSize(),
					Var( TVS_DistributedCoeff ) );
			}
			MathEngine().AllReduce( params[j]->GetData(), params[j]->GetDataSize() );
		}
	}
}

void CDnnSolver::clip( const CObjectArray<CDnnBlob>& paramDiffBlobs )
{
	if( GetVariable( TVS_MinClipping ) <= -FLT_MAX && GetVariable( TVS_MaxClipping ) >= FLT_MAX ) {
		return;
	}
	for( int i = 0; i < paramDiffBlobs.Size(); ++i ) {
		MathEngine().VectorMinMax( paramDiffBlobs[i]->GetData(), paramDiffBlobs[i]->GetData(),
			paramDiffBlobs[i]->GetDataSize(), Var( TVS_MinClipping ), Var( TVS_MaxClipping ) );
	}
}

void CDnnSolver::clipGradients(const CObjectArray<CDnnBlob>& paramDiffBlobs)
{
	if(paramDiffBlobs.Size() == 0) {
		return;
	}

	clip( paramDiffBlobs );

	if( GetVariable( TVS_MaxGradNorm ) < 0 ) {
		return;
	}

	// Calculate the parameter gradient norm
	MathEngine().VectorDotProduct( paramDiffBlobs[0]->GetData(), paramDiffBlobs[0]->GetData(),
		paramDiffBlobs[0]->GetDataSize(), UseVar( TVS_GradVar ) );
	for(int i = 1; i < paramDiffBlobs.Size(); ++i) {
		MathEngine().VectorDotProduct( paramDiffBlobs[i]->GetData(), paramDiffBlobs[i]->GetData(),
			paramDiffBlobs[i]->GetDataSize(), UseVar( TVS_TempVar ) );
		MathEngine().VectorAdd( Var( TVS_GradVar ), Var( TVS_TempVar ), UseVar( TVS_GradVar ), 1 );
	}
	MathEngine().VectorSqrt( Var( TVS_GradVar ), UseVar( TVS_GradVar ), 1 );

	// Calculate scale
	MathEngine().VectorEltwiseMax( Var( TVS_GradVar ), Var( TVS_MaxGradNorm ), UseVar( TVS_GradVar ), 1 );
	MathEngine().VectorEltwiseDivide( Var( TVS_MaxGradNorm ), Var( TVS_GradVar ), UseVar( TVS_TempVar ), 1 );

	// Decrease the gradient
	for(int i = 0; i < paramDiffBlobs.Size(); ++i) {
		MathEngine().VectorMultiply( paramDiffBlobs[i]->GetData(), paramDiffBlobs[i]->GetData(),
			paramDiffBlobs[i]->GetDataSize(), Var( TVS_TempVar ) );
	}
}

static CString concatLayerPath( const CArray<CString>& path )
{
	CString layerPath = path[0];
	for( int i = 1; i < path.Size(); ++i ) {
		layerPath += layerPathSeparator + path[i];
	}
	return layerPath;
}

void CDnnSolver::loadPrevVersionDnnSolverMaps( CArchive& archive, const CDnn& dnn )
{
	CMap<CString, CArray<CString>> layerPrevIdToPath;
	auto mapLayerIdToPath = [&layerPrevIdToPath]( const CDnnLayerGraph& dnn, auto& mapLayerIdToPath ) -> void
	{
		CArray<const char*> layerNames;
		dnn.GetLayerList( layerNames );
		for( const char* layerName : layerNames ) {
			const CBaseLayer* layer = dnn.GetLayer( layerName );
			const CString layerPath = layer->GetPath( "" );
			CArray<CString>& path = layerPrevIdToPath.GetOrCreateValue( layerPath );
			layer->GetPath( path );
			NeoAssert( path.Size() );
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
		return concatLayerPath( path );
	};

	int size;
	archive >> size;
	for( int i = 0; i < size; ++i ) {
		const CBaseLayer* layerTemp = nullptr;
		const CString layerPath = convertOldIdToLayerPath( &layerTemp );

		CDiffBlobSum& blobSum = layerToParamDiffBlobsSum.GetOrCreateValue( layerPath );
		archive >> blobSum.Count;
		SerializeBlobs( mathEngine, archive, blobSum.Sum );
		blobSum.LayerOwner = layerTemp;
	}

	archive >> size;
	for( int i = 0; i < size; ++i ) {
		const CString layerPath = convertOldIdToLayerPath( nullptr );
		SerializeBlobs( mathEngine, archive, layerToGradientHistory.GetOrCreateValue( layerPath ) );
	}
}

static const int DnnSolverVersion = 2;

void CDnnSolver::Serialize( CArchive& archive, const CDnn& dnn )
{
	const int version = archive.SerializeVersion( DnnSolverVersion );
	if( archive.IsStoring() ) {
		archive << layerToParamDiffBlobsSum.Size();
		for( int pos = layerToParamDiffBlobsSum.GetFirstPosition(); pos != NotFound;
			pos = layerToParamDiffBlobsSum.GetNextPosition( pos ) )
		{
			CString layerPath = layerToParamDiffBlobsSum.GetKey( pos );
			const CBaseLayer* layer = layerToParamDiffBlobsSum.GetValue( pos ).LayerOwner;
			NeoAssert( layer != nullptr );
			CArray<CString> path;
			layer->GetPath( path );
			archive.Serialize( path );
			NeoAssert( path.Size() );

			archive << layerToParamDiffBlobsSum.GetValue( pos ).Count;
			SerializeBlobs( mathEngine, archive, layerToParamDiffBlobsSum.GetValue( pos ).Sum );

			const bool hasGradientHistory = layerToGradientHistory.Has( layerPath );
			archive << hasGradientHistory;
			if( hasGradientHistory ) {
				SerializeBlobs( mathEngine, archive, layerToGradientHistory.GetValue( pos ) );
			}
		}
		archive << GetLearningRate() << GetL1Regularization() << GetL2Regularization() << GetMaxGradientNorm();
		archive << GetVariable( TVS_MinClipping ) << GetVariable( TVS_MaxClipping );
	} else {
		layerToParamDiffBlobsSum.DeleteAll();
		layerToGradientHistory.DeleteAll();
		layersToReduce.DeleteAll();
		reduceOrder.DeleteAll();

		if( version >= 2 ) {
			int size;
			archive >> size;
			for( int i = 0; i < size; ++i ) {
				CArray<CString> path;
				archive.Serialize( path );
				NeoAssert( path.Size() );

				const CString layerPath = concatLayerPath( path );
				CDiffBlobSum& blobSum = layerToParamDiffBlobsSum.GetOrCreateValue( layerPath );
				archive >> blobSum.Count;
				SerializeBlobs( mathEngine, archive, blobSum.Sum );
				blobSum.LayerOwner = dnn.GetLayer( path );

				bool hasGradientHistory;
				archive >> hasGradientHistory;
				if( hasGradientHistory ) {
					SerializeBlobs( mathEngine, archive, layerToGradientHistory.GetOrCreateValue( layerPath ) );
				}
			}
		} else {
			loadPrevVersionDnnSolverMaps( archive, dnn );
		}

		float learningRate, regularizationL1, regularizationL2, maxGradientNorm;
		archive >> learningRate >> regularizationL1 >> regularizationL2 >> maxGradientNorm;

		SetLearningRate( learningRate );
		SetL1Regularization( regularizationL1 );
		SetL2Regularization( regularizationL2 );
		SetMaxGradientNorm( maxGradientNorm );

		if( version >= 1 ) {
			float clipGradientMin, clipGradientMax;
			archive >> clipGradientMin >> clipGradientMax;
			SetMinMaxGradientClipping( clipGradientMin, clipGradientMax );
		} else {
			SetMinMaxGradientClipping( /*min*/-FLT_MAX, /*max*/FLT_MAX );
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CDnnSimpleGradientSolver::CDnnSimpleGradientSolver( IMathEngine& mathEngine ) :
	CDnnSolver( mathEngine, TV_Count )
{
	SetMomentDecayRate( 0.9f );
}

static const int DnnSimpleGradientSolverVersion = 0;

void CDnnSimpleGradientSolver::Serialize( CArchive& archive, const CDnn& dnn )
{
	archive.SerializeVersion( DnnSimpleGradientSolverVersion );
	CDnnSolver::Serialize( archive, dnn );

	float momentDecayRate = GetMomentDecayRate();
	archive.Serialize( momentDecayRate );
	archive.Serialize( isInCompatibilityMode );

	if( archive.IsLoading() ) {
		SetMomentDecayRate( momentDecayRate );
	}
}

void CDnnSimpleGradientSolver::SetMomentDecayRate( float decayRate )
{
	SetVariable( TV_MomentDecayRate, decayRate );
	SetVariable( TV_OpMomentDecayRate, 1 - decayRate );
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

	const float rate = layer->GetLearningRate() * GetLearningRate();
	const float regL1 = layer->GetL1RegularizationMult() * GetL1Regularization();
	const float regL2 = layer->GetL2RegularizationMult() * GetL2Regularization();

	// Set the values of the variables
	SetVariable( TV_OpRegL2MomentDecayRate, IsInCompatibilityMode() ? ( ( 1 - GetMomentDecayRate() ) * regL2 ) : ( -rate * regL2 ) );
	SetVariable( TV_Rate, -rate );
	SetVariable( TV_L1Threshold, regL1 );
	SetVariable( TV_L1Mult, IsInCompatibilityMode() ? 1.f : ( -rate ) );

	for( int i = 0; i < paramBlobs.Size(); ++i ) {
		const int dataSize = paramBlobs[i]->GetDataSize();

		// Update the gradient in history
		MathEngine().VectorMultiply( gradientHistory[i]->GetData(),
			gradientHistory[i]->GetData(), dataSize, Var( TV_MomentDecayRate ) );
		MathEngine().VectorMultiplyAndAdd( gradientHistory[i]->GetData(), paramDiffBlobs[i]->GetData(),
			gradientHistory[i]->GetData(), dataSize, Var( IsInCompatibilityMode() ? TV_OpMomentDecayRate : TV_Rate ) );

		if(regL2 > 0) {
			MathEngine().VectorMultiplyAndAdd( gradientHistory[i]->GetData(), paramBlobs[i]->GetData(),
				gradientHistory[i]->GetData(), dataSize, Var( TV_OpRegL2MomentDecayRate ) );
		}
		if(regL1 > 0) {
			MathEngine().VectorL1DiffAdd( gradientHistory[i]->GetData(), paramBlobs[i]->GetData(),
				gradientHistory[i]->GetData(), dataSize, Var( TV_L1Threshold ), Var( TV_L1Mult ) );
		}

		// Add regularization and gradient
		if( isInCompatibilityMode ) {
			MathEngine().VectorMultiplyAndAdd( paramBlobs[i]->GetData(), gradientHistory[i]->GetData(),
				paramBlobs[i]->GetData(), dataSize, Var( TV_Rate ) );
		} else {
			MathEngine().VectorAdd( paramBlobs[i]->GetData(), gradientHistory[i]->GetData(),
				paramBlobs[i]->GetData(), dataSize );
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CDnnAdaptiveGradientSolver::CDnnAdaptiveGradientSolver( IMathEngine& mathEngine ) :
	CDnnSolver( mathEngine, TV_Count ),
	momentDecayRateN( 1.f ),
	secondMomentDecayRateN( 1.f ),
	isAmsGradEnabled( false ),
	isDecoupledWeightDecay( false )
{
	SetMomentDecayRate( 0.9f );
	SetSecondMomentDecayRate( 0.999f );
	SetEpsilon( 1e-6f );
	SetVariable( TV_L1Mult, 1.f );
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

static const int DnnAdaptiveGradientSolver = 1;

void CDnnAdaptiveGradientSolver::Serialize( CArchive& archive, const CDnn& dnn )
{
	const int version = archive.SerializeVersion( DnnAdaptiveGradientSolver );
	CDnnSolver::Serialize( archive, dnn );

	float momentDecayRate = GetMomentDecayRate();
	archive.Serialize( momentDecayRate );
	archive.Serialize( momentDecayRateN );
	float secondMomentDecayRate = GetSecondMomentDecayRate();
	archive.Serialize( secondMomentDecayRate );
	archive.Serialize( secondMomentDecayRateN );
	float epsilon = GetEpsilon();
	archive.Serialize( epsilon );
	archive.Serialize( isAmsGradEnabled );
	if( version < 1 ) {
		isDecoupledWeightDecay = false;
	} else {
		archive.Serialize( isDecoupledWeightDecay );
	}
	archive.Serialize( isInCompatibilityMode );
	if( archive.IsLoading() ) {
		SetMomentDecayRate( momentDecayRate );
		SetSecondMomentDecayRate( secondMomentDecayRate );
		SetEpsilon( epsilon );
	}
}

void CDnnAdaptiveGradientSolver::SetMomentDecayRate( float decayRate )
{
	SetVariable( TV_MomentDecayRate, decayRate );
	SetVariable( TV_OpMomentDecayRate, 1 - decayRate );
}

void CDnnAdaptiveGradientSolver::SetSecondMomentDecayRate( float decayRate )
{
	SetVariable( TV_SecondMomentDecayRate, decayRate );
	SetVariable( TV_OpSecondMomentDecayRate, 1 - decayRate );
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
	momentDecayRateN *= GetMomentDecayRate();
	secondMomentDecayRateN *= GetSecondMomentDecayRate();
}

// Add regularization
static const CDnnBlob* addRegularization( IMathEngine& mathEngine,
	const CDnnBlob* diffBlob, const CDnnBlob* params, float regL1, float regL2,
	const CConstFloatHandle& l1Threshold, const CConstFloatHandle& l1Mult, const CConstFloatHandle& l2Reg,
	CDnnBlob* temporaryBlob )
{
	if( regL2 > 0 ) {
		mathEngine.VectorMultiplyAndAdd( diffBlob->GetData(), params->GetData(),
			temporaryBlob->GetData(), params->GetDataSize(), l2Reg );
		diffBlob = temporaryBlob;
	}
	if( regL1 > 0 ) {
		mathEngine.VectorL1DiffAdd( diffBlob->GetData(), params->GetData(),
			temporaryBlob->GetData(), params->GetDataSize(), l1Threshold, l1Mult );
		diffBlob = temporaryBlob;
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
	SetVariable( TV_RegL2, regL2 );
	SetVariable( TV_Rate, -rate );
	SetVariable( TV_L1Threshold, regL1 );

	for( int i = 0; i < paramBlobs.Size(); ++i ) {
		const int dataSize = paramBlobs[i]->GetDataSize();
		CDnnBlob* moment = gradientHistory[i];
		CDnnBlob* secondMoment = gradientHistory[i + paramDiffBlobs.Size() * GHT_SecondMomentAverage];
		const CDnnBlob* paramDiffBlob = paramDiffBlobs[i];

		if( temporaryBlob == nullptr || temporaryBlob->GetDataSize() < paramDiffBlobs[i]->GetDataSize() ) {
			temporaryBlob = CDnnBlob::CreateVector( MathEngine(), CT_Float, paramDiffBlobs[i]->GetDataSize() );
		}

		if( !IsDecoupledWeightDecay() ) {
			paramDiffBlob = addRegularization( MathEngine(), paramDiffBlob, paramBlobs[i], regL1, regL2,
				Var( TV_L1Threshold ), Var( TV_L1Mult ), Var( TV_RegL2 ), temporaryBlob );
		}

		// Update the historical gradient
		MathEngine().VectorMultiply( moment->GetData(), moment->GetData(), dataSize,
			Var( TV_MomentDecayRate ) );
		MathEngine().VectorMultiplyAndAdd( moment->GetData(), paramDiffBlob->GetData(),
			moment->GetData(), dataSize, Var( TV_OpMomentDecayRate ) );
		// Calculate the historical average squared gradient
		MathEngine().VectorEltwiseMultiply( paramDiffBlob->GetData(), paramDiffBlob->GetData(),
			temporaryBlob->GetData(), dataSize );
		MathEngine().VectorMultiply( secondMoment->GetData(), secondMoment->GetData(), dataSize,
			Var( TV_SecondMomentDecayRate ) );
		MathEngine().VectorMultiplyAndAdd( secondMoment->GetData(), temporaryBlob->GetData(),
			secondMoment->GetData(), dataSize, Var( TV_OpSecondMomentDecayRate ) );
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
		MathEngine().VectorAddValue( temporaryBlob->GetData(), temporaryBlob->GetData(), dataSize, Var( TV_Epsilon ) );
		// Divide the historical gradient by the square root
		MathEngine().VectorEltwiseDivide( moment->GetData(), temporaryBlob->GetData(),
			temporaryBlob->GetData(), dataSize );

		const CDnnBlob* ptrBlob = temporaryBlob;
		if( IsDecoupledWeightDecay() ) {
			ptrBlob = addRegularization( MathEngine(), temporaryBlob, paramBlobs[i], regL1, regL2,
				Var( TV_L1Threshold ), Var( TV_L1Mult ), Var( TV_RegL2 ), temporaryBlob );
		}
		// Add the gradient
		MathEngine().VectorMultiplyAndAdd( paramBlobs[i]->GetData(), ptrBlob->GetData(),
			paramBlobs[i]->GetData(), dataSize, Var( TV_Rate ) );
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CDnnNesterovGradientSolver::CDnnNesterovGradientSolver( IMathEngine& mathEngine ) :
	CDnnSolver( mathEngine, TV_Count ),
	secondMomentDecayRateN( 1.f ),
	isAmsGradEnabled( false ),
	isDecoupledWeightDecay( false ),
	trainCount( 0 ),
	productMuT( 1.f )
{
	SetMomentDecayRate( 0.9f );
	SetSecondMomentDecayRate( 0.99f );
	SetEpsilon( 1e-6f );
	SetVariable( TV_L1Mult, 1.f );
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

static const int DnnNesterovGradientSolverVersion = 1;

void CDnnNesterovGradientSolver::Serialize( CArchive& archive, const CDnn& dnn )
{
	const int version = archive.SerializeVersion( DnnNesterovGradientSolverVersion );
	CDnnSolver::Serialize( archive, dnn );

	float momentDecayRate = GetMomentDecayRate();
	archive.Serialize( momentDecayRate );
	float secondMomentDecayRate = GetSecondMomentDecayRate();
	archive.Serialize( secondMomentDecayRate );
	archive.Serialize( secondMomentDecayRateN );
	float epsilon = GetEpsilon();
	archive.Serialize( epsilon );
	archive.Serialize( isAmsGradEnabled );
	if( version < 1 ) {
		isDecoupledWeightDecay = false;
	} else {
		archive.Serialize( isDecoupledWeightDecay );
	}
	archive.Serialize( trainCount );
	archive.Serialize( productMuT );

	if( archive.IsLoading() ) {
		SetMomentDecayRate( momentDecayRate );
		SetSecondMomentDecayRate( secondMomentDecayRate );
		SetEpsilon( epsilon );
	}
}

void CDnnNesterovGradientSolver::SetMomentDecayRate( float decayRate )
{
	SetVariable( TV_MomentDecayRate, decayRate );
	SetVariable( TV_OpMomentDecayRate, 1 - decayRate );
}

void CDnnNesterovGradientSolver::SetSecondMomentDecayRate( float decayRate )
{
	SetVariable( TV_SecondMomentDecayRate, decayRate );
	SetVariable( TV_OpSecondMomentDecayRate, 1 - decayRate );
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
	secondMomentDecayRateN *= GetSecondMomentDecayRate();
	// The "magic numbers" are from the reference paper
	trainCount++;
	muT = GetMomentDecayRate() * ( 1 - 0.5f * powf( 0.96f, trainCount * 0.004f ) );
	muTPlusOne = GetMomentDecayRate() * ( 1 - 0.5f * powf( 0.96f, ( trainCount + 1 ) * 0.004f ) );
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
	const float rate = layer->GetLearningRate() * GetLearningRate();
	const float regL1 = layer->GetL1RegularizationMult() * GetL1Regularization();
	const float regL2 = layer->GetL2RegularizationMult() * GetL2Regularization();

	// Set the values for the variables
	SetVariable( TV_RegL2, regL2 );
	SetVariable( TV_Rate, -rate );
	SetVariable( TV_L1Threshold, regL1 );
	SetVariable( TV_MBarGradMult, ( 1.f - muT ) / ( 1.f - productMuT ) );
	SetVariable( TV_MBarMomentMult, muTPlusOne / ( 1.f - productMuT * muTPlusOne ) );
	SetVariable( TV_InvOpSecondMomentDecayRateN, 1.f / ( 1 - secondMomentDecayRateN ) );

	for( int i = 0; i < paramBlobs.Size(); ++i ) {
		const int dataSize = paramBlobs[i]->GetDataSize();
		CDnnBlob* moment = gradientHistory[i];
		CDnnBlob* secondMoment = gradientHistory[i + paramDiffBlobs.Size() * GHT_SecondMomentAverage];
		const CDnnBlob* paramDiffBlob = paramDiffBlobs[i];

		if( temporaryBlob == nullptr || temporaryBlob->GetDataSize() < paramDiffBlobs[i]->GetDataSize() ) {
			temporaryBlob = CDnnBlob::CreateVector( MathEngine(), CT_Float, paramDiffBlobs[i]->GetDataSize() );
			mBarBlob = temporaryBlob->GetClone();
		}

		if( !IsDecoupledWeightDecay() ) {
			paramDiffBlob = addRegularization( MathEngine(), paramDiffBlob, paramBlobs[i], regL1, regL2,
				Var( TV_L1Threshold ), Var( TV_L1Mult ), Var( TV_RegL2 ), temporaryBlob );
		}

		// Update the historical gradient
		MathEngine().VectorMultiply( moment->GetData(), moment->GetData(), dataSize,
			Var( TV_MomentDecayRate ) );
		MathEngine().VectorMultiplyAndAdd( moment->GetData(), paramDiffBlob->GetData(),
			moment->GetData(), dataSize, Var( TV_OpMomentDecayRate ) );
		// Calculate the historical average squared gradient
		MathEngine().VectorEltwiseMultiply( paramDiffBlob->GetData(), paramDiffBlob->GetData(),
			temporaryBlob->GetData(), dataSize );
		MathEngine().VectorMultiply( secondMoment->GetData(), secondMoment->GetData(), dataSize,
			Var( TV_SecondMomentDecayRate ) );
		MathEngine().VectorMultiplyAndAdd( secondMoment->GetData(), temporaryBlob->GetData(),
			secondMoment->GetData(), dataSize, Var( TV_OpSecondMomentDecayRate ) );

		// Calculate the auxiliary variables (notations taken from the reference paper)
		// m with a dash
		CFloatHandle mBar = mBarBlob->GetData();
		MathEngine().VectorMultiply( paramDiffBlob->GetData(), mBar, dataSize, Var( TV_MBarGradMult ) );
		MathEngine().VectorMultiplyAndAdd( mBar, moment->GetData(), mBar, dataSize, Var( TV_MBarMomentMult ) );

		// sqrt(n with a hat) + eps
		if( IsAmsGradEnabled() ) {
			// Update the maximum average
			CDnnBlob* secondMomentMaxAverage = gradientHistory[i + paramDiffBlobs.Size() * GHT_SecondMomentMaxAverage];
			MathEngine().VectorEltwiseMax( secondMomentMaxAverage->GetData(), secondMoment->GetData(),
				secondMomentMaxAverage->GetData(), secondMomentMaxAverage->GetDataSize() );
			// n with a hat calculated for the maximum of the second moment moving mean
			MathEngine().VectorMultiply( secondMomentMaxAverage->GetData(), temporaryBlob->GetData(), dataSize,
				Var( TV_InvOpSecondMomentDecayRateN ) );
		} else {
			// n with a hat calculated for the second momentum moving mean
			MathEngine().VectorMultiply( secondMoment->GetData(), temporaryBlob->GetData(), dataSize,
				Var( TV_InvOpSecondMomentDecayRateN ) );
		}
		MathEngine().VectorSqrt( temporaryBlob->GetData(), temporaryBlob->GetData(), dataSize );
		MathEngine().VectorAddValue( temporaryBlob->GetData(), temporaryBlob->GetData(), dataSize, Var( TV_Epsilon ) );
		// Calculate the final diff
		MathEngine().VectorEltwiseDivide( mBar, temporaryBlob->GetData(), temporaryBlob->GetData(), dataSize );

		const CDnnBlob* ptrBlob = temporaryBlob;
		if( IsDecoupledWeightDecay() ) {
			ptrBlob = addRegularization( MathEngine(), temporaryBlob, paramBlobs[i], regL1, regL2,
				Var( TV_L1Threshold ), Var( TV_L1Mult ), Var( TV_RegL2 ), temporaryBlob );
		}
		// Update parameters
		MathEngine().VectorMultiplyAndAdd( paramBlobs[i]->GetData(), ptrBlob->GetData(),
			paramBlobs[i]->GetData(), dataSize, Var( TV_Rate ) );
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CDnnLambGradientSolver::CDnnLambGradientSolver( IMathEngine& mathEngine ) :
	CDnnSolver( mathEngine, TV_Count ),
	useTrustRatio( true ),
	useNvLamb( false ),
	totalGradientNorm( 1.0f )
{
	SetL2Regularization( 0.01f );
	// External config variables
	SetMomentDecayRate( 0.9f );
	SetSecondMomentDecayRate( 0.999f );
	SetEpsilon( 1e-6f );
	SetWeightDecayClip( -1.f );
	// Internal calculations variables
	SetVariable( TV_LayerNorm, 0.f );
	SetVariable( TV_TrustRatio, 0.f );
	SetVariable( TV_L2WeightNorm, 0.f );
	SetVariable( TV_L2UpdateNorm, 0.f );
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

static const int DnnLambGradientSolverVersion = 0;

void CDnnLambGradientSolver::Serialize( CArchive& archive, const CDnn& dnn )
{
	archive.SerializeVersion( DnnLambGradientSolverVersion );
	CDnnSolver::Serialize( archive, dnn );

	float momentDecayRate = GetMomentDecayRate();
	archive.Serialize( momentDecayRate );
	float secondMomentDecayRate = GetSecondMomentDecayRate();
	archive.Serialize( secondMomentDecayRate );
	float epsilon = GetEpsilon();
	archive.Serialize( epsilon );
	float weightDecayClip = GetWeightDecayClip();
	archive.Serialize( weightDecayClip );
	archive.Serialize( useTrustRatio );
	archive.Serialize( useNvLamb );
	archive.Serialize( layersGradientNormSquare );

	int excludedLayersCount = excludedLayers.Size();
	archive.Serialize( excludedLayersCount );

	if( archive.IsLoading() ) {
		SetMomentDecayRate( momentDecayRate );
		SetSecondMomentDecayRate( secondMomentDecayRate );
		SetEpsilon( epsilon );
		SetWeightDecayClip( weightDecayClip );
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

void CDnnLambGradientSolver::SetMomentDecayRate( float decayRate )
{
	SetVariable( TV_MomentDecayRate, decayRate );
	SetVariable( TV_OpMomentDecayRate, 1 - decayRate );
}

void CDnnLambGradientSolver::SetSecondMomentDecayRate( float decayRate )
{
	SetVariable( TV_SecondMomentDecayRate, decayRate );
	SetVariable( TV_OpSecondMomentDecayRate, 1 - decayRate );
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
	const float rate = layer->GetLearningRate() * GetLearningRate();
	const float layerWeighDecay = GetL2Regularization() * layer->GetL2RegularizationMult();
	const float clipMultiplier = 1.0f / max( 1.0f, totalGradientNorm );

	// Set the values for the variables
	SetVariable( TV_Rate, -rate );
	SetVariable( TV_WeightDecay, layerWeighDecay );
	SetVariable( TV_ClipMultiplier, clipMultiplier );

	// Getting parameters affected by weight decay
	CHashTable<int> weightDecayParamIndexes;
	getWeightDecayIndices( *layer, paramBlobs.Size(), weightDecayParamIndexes );

	for( int i = 0; i < paramBlobs.Size(); ++i ) {
		const int dataSize = paramBlobs[i]->GetDataSize();
		CDnnBlob* moment = gradientHistory[i];
		CDnnBlob* secondMoment = gradientHistory[i + paramDiffBlobs.Size() * GHT_SecondMomentAverage];

		if( tempBlob == nullptr || tempBlob->GetDataSize() != paramDiffBlobs[i]->GetDataSize() ) {
			tempBlob = CDnnBlob::CreateVector( MathEngine(), CT_Float, paramDiffBlobs[i]->GetDataSize() );
		}

		CPtr<CDnnBlob> paramDiffBlob = paramDiffBlobs[i];

		if( useNvLamb ) {
			MathEngine().VectorMultiply( paramDiffBlob->GetData(), paramDiffBlob->GetData(), dataSize,
				Var( TV_ClipMultiplier ) );
		}

		// Update the historical gradient
		MathEngine().VectorMultiply( moment->GetData(), moment->GetData(), dataSize, Var( TV_MomentDecayRate ) );
		MathEngine().VectorMultiplyAndAdd( moment->GetData(), paramDiffBlob->GetData(),
			moment->GetData(), dataSize, Var( TV_OpMomentDecayRate ) );

		// Calculate the historical average squared gradient
		MathEngine().VectorEltwiseMultiply( paramDiffBlob->GetData(), paramDiffBlob->GetData(),
			tempBlob->GetData(), dataSize );

		// Add squared L2-norm for calculation of L2-norm of the whole mode
		if( useNvLamb ) {
			const float invSquareClipMultiplier = 1.0f / ( clipMultiplier * clipMultiplier );
			MathEngine().VectorFill( UseVar( TV_LayerNorm ), 0.f, 1 );
			MathEngine().VectorSum( tempBlob->GetData(), dataSize, UseVar( TV_LayerNorm ) );
			layersGradientNormSquare.Add( invSquareClipMultiplier * Var( TV_LayerNorm ).GetValue() ); // CUDA sync
		}

		MathEngine().VectorMultiply( secondMoment->GetData(), secondMoment->GetData(), dataSize,
			Var( TV_SecondMomentDecayRate ) );
		MathEngine().VectorMultiplyAndAdd( secondMoment->GetData(), tempBlob->GetData(),
			secondMoment->GetData(), dataSize, Var( TV_OpSecondMomentDecayRate ) );

		// square root of the second moment
		MathEngine().VectorSqrt( secondMoment->GetData(), tempBlob->GetData(), dataSize );

		// add epsilon before division
		MathEngine().VectorAddValue( tempBlob->GetData(), tempBlob->GetData(), dataSize, Var( TV_Epsilon ) );

		// divide historical gradient by the square root
		MathEngine().VectorEltwiseDivide( moment->GetData(), tempBlob->GetData(),
			tempBlob->GetData(), dataSize );

		// weightDecay
		if( weightDecayParamIndexes.Has( i ) && layerWeighDecay > 0 ) {
			MathEngine().VectorMultiplyAndAdd( tempBlob->GetData(), paramBlobs[i]->GetData(),
				tempBlob->GetData(), tempBlob->GetDataSize(), Var( TV_WeightDecay ) );
		}

		if( useTrustRatio ) {
			// apply normalizing multiplier
			calcNormalizeMultiplier( *paramBlobs[i], *tempBlob, UseVar( TV_TrustRatio ) );
			MathEngine().VectorMultiply( tempBlob->GetData(), tempBlob->GetData(), dataSize, Var( TV_TrustRatio ) );
		}

		// adding gradient
		MathEngine().VectorMultiplyAndAdd( paramBlobs[i]->GetData(), tempBlob->GetData(),
			paramBlobs[i]->GetData(), dataSize, Var( TV_Rate ) );
	}
}

// L2 norm of a vector devided by vector size.
void CDnnLambGradientSolver::calcL2NormAverage( const CConstFloatHandle& data, int dataSize, int normId ) const
{
	NeoAssert( dataSize > 0 );
	NeoAssert( normId >= TV_L2WeightNorm && normId <= TV_L2UpdateNorm );

	CPtr<CDnnBlob> temp = CDnnBlob::CreateVector( MathEngine(), CT_Float, dataSize );
	MathEngine().VectorFill( UseVar( normId ), 1.f / dataSize, 1 );
	MathEngine().VectorMultiply( data, temp->GetData(), dataSize, Var( normId ) );

	MathEngine().VectorFill( UseVar( normId ), 0.f, 1 );
	MathEngine().VectorDotProduct( temp->GetData(), temp->GetData(), dataSize, UseVar( normId ) );
	MathEngine().VectorSqrt( Var( normId ), UseVar( normId ), 1 );
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
void CDnnLambGradientSolver::calcNormalizeMultiplier( const CDnnBlob& weights, const CDnnBlob& update,
	const CFloatHandle& multiplier ) const
{
	calcL2NormAverage( weights.GetData(), weights.GetDataSize(), TV_L2WeightNorm );
	if( GetVariable( TV_WeightDecayClip ) > 0 ) {
		MathEngine().VectorEltwiseMin( Var( TV_L2WeightNorm ), Var( TV_WeightDecayClip ), UseVar( TV_L2WeightNorm ), 1 );
	}
	calcL2NormAverage( update.GetData(), update.GetDataSize(), TV_L2UpdateNorm );

	MathEngine().VectorEltwiseMin( Var( TV_L2WeightNorm ), Var( TV_L2UpdateNorm ), multiplier, 1 );
	if( multiplier.GetValue() > 0 ) { // CUDA sync
		MathEngine().VectorEltwiseDivide( Var( TV_L2WeightNorm ), Var( TV_L2UpdateNorm ), multiplier, 1 );
	} else {
		MathEngine().VectorFill( multiplier, 1.f, 1 );
	}
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
	if( totalGradientNorm < GetEpsilon() ) {
		totalGradientNorm = 1.0f;
	}
	layersGradientNormSquare.Empty();
}

} // namespace NeoML
