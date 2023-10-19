/* Copyright © 2023 ABBYY

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

#include <NeoML/Dnn/Layers/LoraFullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include <NeoML/Dnn/DnnDistributed.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

static const char* const baseFcName = "FullyConnectedBase";
static const char* const loraFcAName = "FullyConnectedALoRA";
static const char* const loraFcBName = "FullyConnectedBLoRA";
static const char* const loraDropoutName = "DropoutLoRA";
static const char* const loraScalingName = "ScalingLoRA";
static const char* const loraSumName = "SumLoRA";

//----------------------------------------------------------------------------------------------

CLoraFullyConnectedLayer::CLoraFullyConnectedLayer( IMathEngine& mathEngine, const char* name ) :
	CCompositeLayer( mathEngine, ( name == nullptr ) ? "CLoraFullyConnectedLayer" : name )
{
	buildLayer();
}

CLoraFullyConnectedLayer::CLoraFullyConnectedLayer( CFullyConnectedLayer& fc ) :
	CCompositeLayer( fc.MathEngine(), fc.GetName() ),
	baseFc( &fc )
{
	baseFc->SetName( baseFcName );
	SetInputMapping( *baseFc );
	SetOutputMapping( *baseFc );
	AddLayer( *baseFc );

	NeoPresume( baseFc != nullptr && baseFc->GetName() == CString( baseFcName ) );
}

static const int loraFullyConnectedLayerVersion = 0;

// Converted FullyConnectedLayer serialization
void CLoraFullyConnectedLayer::Serialize( CArchive& archive )
{
	( void ) archive.SerializeVersion( loraFullyConnectedLayerVersion );
	archive.Serialize( loraStoreSeparate );

	if( loraStoreSeparate == true ) {
		CheckArchitecture( loraRank == 0, GetPath(),
			"Should contains only baseFc, please mergeLora or destroyLora before call" );
	}
	CCompositeLayer::Serialize( archive );

	archive.Serialize( loraRank );
	archive.Serialize( loraAlpha );
	archive.Serialize( loraDropoutRate );

	if( archive.IsLoading() ) {
		baseFc = CheckCast<CFullyConnectedLayer>( GetLayer( baseFcName ) );
		// optional LoRA layers
		loraFcA = HasLayer( loraFcAName )
			? CheckCast<CFullyConnectedLayer>( GetLayer( loraFcAName ) )
			: nullptr;

		if( loraFcA != nullptr ) {
			NeoAssert( loraRank != 0 && loraAlpha != 0 && loraStoreSeparate == false );

			loraFcB = CheckCast<CFullyConnectedLayer>( GetLayer( loraFcBName ) );
			loraDropout = HasLayer( loraDropoutName )
				? CheckCast<CDropoutLayer>( GetLayer( loraDropoutName ) )
				: nullptr;
			NeoAssert( loraDropout == nullptr || loraDropoutRate > 0 );
			loraScaling = CheckCast<CLinearLayer>( GetLayer( loraScalingName ) );
			loraSum = CheckCast<CEltwiseSumLayer>( GetLayer( loraSumName ) );
		} else {
			NeoAssert( loraRank == 0 && loraAlpha == 0 && loraDropoutRate == 0 );
			// Otherwise, it isn't built
			loraFcB = nullptr;
			loraDropout = nullptr;
			loraScaling = nullptr;
			loraSum = nullptr;
		}
	}
}

void CLoraFullyConnectedLayer::SerializeWeightsLoRA( CArchive& archive )
{
	NeoAssert( loraStoreSeparate == true );
	NeoAssert( baseFc != nullptr );

	if( archive.IsLoading() ) {
		int rank = 0;
		archive.Serialize( rank );
		float alpha = 0;
		archive.Serialize( alpha );
		float dropoutRate = 0;
		archive.Serialize( dropoutRate );

		DestroyUnmergedLoRA();
		BuildLoRA( rank, alpha, dropoutRate );
	} else { // storing
		NeoAssert( loraRank != 0 && loraAlpha != 0 );
		archive.Serialize( loraRank );
		archive.Serialize( loraAlpha );
		archive.Serialize( loraDropoutRate );
	}

	CPtr<CDnnBlob> aWeights = AWeightsLoRA(); // No copy
	SerializeBlob( MathEngine(), archive, aWeights );
	SetAWeightsLoRAData( aWeights );

	CPtr<CDnnBlob> bWeights = BWeightsLoRA(); // No copy
	SerializeBlob( MathEngine(), archive, bWeights );
	SetBWeightsLoRAData( bWeights );
}

void CLoraFullyConnectedLayer::buildLayer()
{
	baseFc = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	baseFc->SetName( baseFcName );
	baseFc->SetNumberOfElements( 1 );
	SetInputMapping( *baseFc );
	SetOutputMapping( *baseFc );
	AddLayer( *baseFc );

	NeoPresume( baseFc != nullptr && baseFc->GetName() == CString( baseFcName ) );
}

void CLoraFullyConnectedLayer::Reshape()
{
	CheckLayerArchitecture( GetInputCount() == 1, "Layer must have only 1 input" );
	const bool uninitializedLora = ( loraFcB != nullptr && loraFcB->Weights() == nullptr );

	CCompositeLayer::Reshape();

	// LoRA A == Gaussian initialization, it is Xavier, already by default
	if( loraFcB != nullptr && uninitializedLora == true ) {
		// LoRA B == Zero initialization
		loraFcB->Weights()->Fill( 0.f );
	}
}

float CLoraFullyConnectedLayer::GetDropoutRateLoRA() const
{
	return ( loraDropout == nullptr )
		? loraDropoutRate
		: loraDropout->GetDropoutRate();
}

void CLoraFullyConnectedLayer::SetAWeightsLoRAData( const CDnnBlob* weights )
{
	NeoAssert( loraFcA != nullptr );
	loraFcA->SetWeightsData( weights );
}

void CLoraFullyConnectedLayer::SetBWeightsLoRAData( const CDnnBlob* weights )
{
	NeoAssert( loraFcB != nullptr );
	loraFcB->SetWeightsData( weights );
}

void CLoraFullyConnectedLayer::BuildLoRA( int rank, float alpha, float dropoutRate )
{
	NeoAssert( loraFcA == nullptr ); // check that LoRA hasn't been built yet

	NeoPresume( loraFcB == nullptr && loraDropout == nullptr );
	NeoPresume( loraScaling == nullptr && loraSum == nullptr );
	NeoPresume( loraRank == 0 && loraAlpha == 0 && loraDropoutRate == 0 );

	// Disable update base weights matrix
	baseFc->DisableLearning();

	NeoAssert( rank > 0 && rank <= baseFc->GetNumberOfElements() );
	NeoAssert( alpha > 0.f );
	NeoAssert( dropoutRate < 1.f );

	loraRank = rank;
	loraAlpha = alpha;
	loraDropoutRate = dropoutRate;

	// The same data used for both baseFC and LoRA inputs

	if( dropoutRate > 0.f ) {
		loraDropout = FINE_DEBUG_NEW CDropoutLayer( MathEngine() );
		loraDropout->SetName( loraDropoutName );
		loraDropout->SetDropoutRate( dropoutRate );
		SetInputMapping( *loraDropout );
		AddLayer( *loraDropout );

		NeoPresume( loraDropout != nullptr && loraDropout->GetName() == CString( loraDropoutName ) );
	}

	loraFcA = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	loraFcA->SetName( loraFcAName );
	loraFcA->SetZeroFreeTerm( true ); // should be zero
	loraFcA->SetNumberOfElements( loraRank );
	if( loraDropout != nullptr ) {
		loraFcA->Connect( *loraDropout );
	} else {
		SetInputMapping( *loraFcA );
	}
	AddLayer( *loraFcA );

	NeoPresume( loraFcA != nullptr && loraFcA->GetName() == CString( loraFcAName ) );

	loraFcB = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	loraFcB->SetName( loraFcBName );
	loraFcB->SetZeroFreeTerm( true ); // should be zero
	loraFcB->SetNumberOfElements( baseFc->GetNumberOfElements() );
	loraFcB->Connect( *loraFcA );
	AddLayer( *loraFcB );

	NeoPresume( loraFcB != nullptr && loraFcB->GetName() == CString( loraFcBName ) );

	const float scaling = loraAlpha / loraRank;
	
	loraScaling = FINE_DEBUG_NEW CLinearLayer( MathEngine() );
	loraScaling->SetName( loraScalingName );
	loraScaling->SetMultiplier( scaling );
	loraScaling->SetFreeTerm( 0.f );
	loraScaling->Connect( *loraFcB );
	AddLayer( *loraScaling );
	
	NeoPresume( loraScaling != nullptr && loraScaling->GetName() == CString( loraScalingName ) );

	loraSum = FINE_DEBUG_NEW CEltwiseSumLayer( MathEngine() );
	loraSum->SetName( loraSumName );
	loraSum->Connect( 0, *baseFc );
	loraSum->Connect( 1, *loraScaling );
	AddLayer( *loraSum );

	NeoPresume( loraSum != nullptr && loraSum->GetName() == CString( loraSumName ) );

	// return sum of data instead of the results of baseFC only
	SetOutputMapping( *loraSum );
}

void CLoraFullyConnectedLayer::MergeWeightsLoRA()
{
	NeoAssert( loraFcA != nullptr && loraFcB != nullptr );
	NeoAssert( loraScaling != nullptr && loraSum != nullptr );

	if( loraFcA->Weights() != nullptr ) { // if there was learning

		const CBlobDesc weightsBDesc = loraFcB->Weights()->GetDesc();
		const CBlobDesc baseDesc = baseFc->Weights()->GetDesc();
		const int sizeB = weightsBDesc.BlobSize();
		const int objectSizeB = weightsBDesc.ObjectSize();
		const int objectCountB = weightsBDesc.ObjectCount();
		const int objectSizeA = loraFcA->Weights()->GetDesc().ObjectSize();

		// Memory allocation
		CFloatHandleStackVar temp( MathEngine(), sizeB + sizeof( float ) );
		const CFloatHandle transposeB = temp.GetHandle();
		const CFloatHandle multiplier = temp.GetHandle() + sizeB;
		multiplier.SetValue( loraScaling->GetMultiplier() );

		// ( scale * A * B + W )   = ( W* )
		// ( scale * A * B + W )^T = ( W* )^T
		// scale * B^T * A^T + W^T = ( W* )^T

		// B^T *= scale
		MathEngine().VectorMultiply( /*from*/loraFcB->Weights()->GetData(),
			/*to*/loraFcB->Weights()->GetData(), sizeB, multiplier );

		// B = ( B^T )^T
		MathEngine().TransposeMatrix( /*batchSize*/1, loraFcB->Weights()->GetData(),
			/*height*/objectCountB, /*medium*/1, /*width*/objectSizeB, /*channels*/1,
			transposeB, sizeB );

		// ( W* )^T += B * A^T
		MathEngine().MultiplyTransposedMatrixByMatrixAndAdd(
			transposeB, objectSizeB, objectCountB, objectCountB,
			loraFcA->Weights()->GetData(), objectSizeA, objectSizeA,
			baseFc->Weights()->GetData(), baseDesc.ObjectSize(), baseDesc.BlobSize(),
			nullptr );
	}

	destroyLoRA();
}

void CLoraFullyConnectedLayer::destroyLoRA()
{
	if( loraFcA == nullptr ) { // no exist
		NeoPresume( loraFcB == nullptr && loraSum == nullptr && loraDropout == nullptr );
		NeoAssert( loraScaling == nullptr && loraSum == nullptr );
		return;
	}

	NeoPresume( loraFcA != nullptr && loraFcB != nullptr );
	NeoPresume( loraScaling != nullptr && loraSum != nullptr );

	// Enable update base weights matrix
	baseFc->EnableLearning();

	loraRank = 0;
	loraAlpha = 0;
	loraDropoutRate = 0;

	DeleteLayer( *loraFcA );
	loraFcA = nullptr;
	NeoPresume( loraFcA == nullptr && !HasLayer( loraFcAName ) );

	DeleteLayer( *loraFcB );
	loraFcB = nullptr;
	NeoPresume( loraFcB == nullptr && !HasLayer( loraFcBName ) );

	if( loraDropout != nullptr ) {
		DeleteLayer( *loraDropout );
		loraDropout = nullptr;
		NeoPresume( loraDropout == nullptr && !HasLayer( loraDropoutName ) );
	}

	DeleteLayer( *loraScaling );
	loraScaling = nullptr;
	NeoPresume( loraScaling == nullptr && !HasLayer( loraScalingName ) );

	DeleteLayer( *loraSum );
	loraSum = nullptr;
	NeoPresume( loraSum == nullptr && !HasLayer( loraSumName ) );

	// Revert the output to baseFC only
	SetOutputMapping( *baseFc );
}

//----------------------------------------------------------------------------------------------

CLayerWrapper<CLoraFullyConnectedLayer> LoraFullyConnected( int numberOfElements, bool isZeroFreeTerm )
{
	return CLayerWrapper<CLoraFullyConnectedLayer>( "LoraFullyConnected", [=]( CLoraFullyConnectedLayer* result ) {
		result->SetNumberOfElements( numberOfElements );
		result->SetZeroFreeTerm( isZeroFreeTerm );
	} );
}

//----------------------------------------------------------------------------------------------

void LoraExceptSwitchLearnings( CDnnLayerGraph& graph, bool enable )
{
	auto impl = [enable]( CDnnLayerGraph& graph, auto&& impl ) -> void
	{
		CArray<const char*> layerNames;
		graph.GetLayerList( layerNames );

		for( const char* layerName : layerNames ) {
			CBaseLayer* layer = graph.GetLayer( layerName ).Ptr();

			ILowRankAdapted* loraFc = dynamic_cast<ILowRankAdapted*>( layer );
			if( loraFc != nullptr ) {
				continue; // skip any lora layers
			}

			CCompositeLayer* composite = dynamic_cast<CCompositeLayer*>( layer );
			if( composite != nullptr ) {
				impl( *composite, impl );
				continue; // skip any composite layers
			}

			if( enable ) {
				layer->EnableLearning();
			} else {
				layer->DisableLearning();
			}
		}
	};

	impl( graph, impl );
}

//----------------------------------------------------------------------------------------------

int CLoraSerializer::storeLora( CDnn& dnn, CArchive& archive )
{
	NeoAssert( archive.IsStoring() );
	( void ) archive.SerializeVersion( loraSerializerVersion );

	CArray<CString> layerPath; // recurrenting accumulation

	auto impl = [&archive, &layerPath]( CDnnLayerGraph& graph, auto&& impl ) -> int
	{
		CArray<const char*> layerNames;
		graph.GetLayerList( layerNames );

		int loraLayerCount = 0;
		for( const char* layerName : layerNames ) {
			CBaseLayer* layer = graph.GetLayer( layerName ).Ptr();

			static_assert( LLT_Count == 2, "LLT_Count != 2, LLT_FullyConnected + LLT_End" );
			// LoRA layer is inherited of CCompositeLayer, check first
			CLoraFullyConnectedLayer* loraFc = dynamic_cast<CLoraFullyConnectedLayer*>( layer );
			if( loraFc != nullptr ) {
				// For now only fully-connected is supported but in future embedding will probably be supported
				int loraType = LLT_FullyConnected;
				archive.Serialize( loraType );

				// Store full path to this layer
				// We don't use CBaseLayer::GetPath() here because we may get indistinguashable paths
				// when path separator is used inside of layer names (which sometimes the case esp for exported nets)
				// e.g. "ROOT" / "BLOCK0" / "FC0" vs. "ROOT/BLOCK0" / "FC0" both have path "ROOT/BLOCK0/FC0"
				layerPath.Add( loraFc->GetName() );
				archive.Serialize( layerPath );
				layerPath.DeleteLast();

				loraFc->SetStoreSeparateLoRA( true );
				loraFc->SerializeWeightsLoRA( archive );

				++loraLayerCount;
				continue;
			}

			CCompositeLayer* composite = dynamic_cast<CCompositeLayer*>( layer );
			if( composite != nullptr ) {
				// Updating current path and make recurrent call
				layerPath.Add( composite->GetName() );
				loraLayerCount += impl( *composite, impl );
				layerPath.DeleteLast();
			}
		}
		return loraLayerCount;
	};

	const int loraLayerCount = impl( dnn, impl ); // Store weights and calculate the amount of LoRA layers

	int loraType = LLT_End;
	archive.Serialize( loraType );
	return loraLayerCount;
}

int CLoraSerializer::loadLora( CDnn& dnn, CArchive& archive )
{
	NeoAssert( archive.IsLoading() );
	( void ) archive.SerializeVersion( loraSerializerVersion );

	int loraLayerCount = 0;
	while( true ) {
		int layerType = LLT_Count;
		archive.Serialize( layerType );
		if( layerType == LLT_End ) {
			return loraLayerCount;
		}
		check( layerType < LLT_End, ERR_BAD_ARCHIVE, archive.Name() ); // Check LoRA supported layers only

		CArray<CString> layerPath;
		archive.Serialize( layerPath );

		CDnnLayerGraph* graph = &dnn;
		// Iterating from root dnn over all the composites
		for( int i = 0; i < layerPath.Size() - 1; ++i ) {
			CCompositeLayer* composite = dynamic_cast<CCompositeLayer*>( graph->GetLayer( layerPath[i] ).Ptr() );
			check( composite != nullptr, ERR_BAD_ARCHIVE, archive.Name() ); // Intermediate layer in path must be composite
			graph = composite;
		}
		check( layerPath.Size() >= 1, ERR_BAD_ARCHIVE, archive.Name() ); // Path must contain at least name of last layer

		static_assert( LLT_Count == 2, "LLT_Count != 2, LLT_FullyConnected + LLT_End" );
		// Now LoRA is only supported FullyConnectedLayer
		CLoraFullyConnectedLayer* loraFc = dynamic_cast<CLoraFullyConnectedLayer*>( graph->GetLayer( layerPath.Last() ).Ptr() );
		check( loraFc != nullptr, ERR_BAD_ARCHIVE, archive.Name() ); // Mismatch between CDnn and LoRA weights (layer is missing)

		loraFc->SetStoreSeparateLoRA( true );
		loraFc->SerializeWeightsLoRA( archive );
		++loraLayerCount;
	}
}

int CLoraSerializer::Serialize( CDnn& dnn, CArchive& archive )
{
	if( archive.IsStoring() ) {
		return storeLora( dnn, archive );
	} else if( archive.IsLoading() ) {
		return loadLora( dnn, archive );
	}

	NeoAssert( false );
	return 0;
}

int CLoraSerializer::Serialize( CDistributedTraining& distributed, CArchive& archive )
{
	NeoAssert( distributed.GetModelCount() > 0 );
	if( archive.IsStoring() ) {
		// Inside distributed weights are in sync so we need to serialize only one of them
		return storeLora( *distributed.cnns[0], archive );
	} else if( archive.IsLoading() ) {
		// Load LoRA weights into every net inside distributed
		const int64_t pos = archive.GetPosition();
		const int loadedLayers = loadLora( *distributed.cnns[0], archive );
		for( int i = 1; i < distributed.cnns.Size(); ++i ) {
			archive.Seek( pos, CBaseFile::begin );
			check( loadLora( *distributed.cnns[i], archive ) == loadedLayers, ERR_BAD_ARCHIVE, archive.Name() );
		}
		return loadedLayers;
	}

	NeoAssert( false );
	return 0;
}

int CLoraSerializer::SerializeCheckpoint( CDnn& dnn, CArchive& archive )
{
	// Serializing LoRA weights
	const int result = Serialize( dnn, archive );
	// Serialize solver
	CPtr<CDnnSolver> solverPtr = nullptr;
	if( archive.IsStoring() ) {
		solverPtr = dnn.GetSolver();
	}
	SerializeSolver( archive, dnn, solverPtr );
	if( archive.IsLoading() ) {
		dnn.SetSolver( solverPtr );
	}
	return result;
}

} // namespace NeoML
