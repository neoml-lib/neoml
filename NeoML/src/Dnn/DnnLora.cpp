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

#include <NeoML/Dnn/DnnLora.h>
#include <NeoML/Dnn/DnnDistributed.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/LoraFullyConnectedLayer.h>

namespace NeoML {

void CLoraParams::Serialize( CArchive& archive )
{
	archive.Serialize( Rank );
	archive.Serialize( Alpha );
	archive.Serialize( Dropout );
}

//----------------------------------------------------------------------------------------------------------------------

CLoraBuilder::CLoraBuilder()
{
	// Default layers supported by LoRA
	compositeClasses.Add( { "NeoMLDnnTransformerEncoderLayer", "NeoMLDnnMultiheadAttentionLayer",
		"FmlCnnCompositeLayer", "FmlCnnRecurrentLayer", "FmlCnnTemplateLayer", "NeoMLTemplateLayerExt" } );
}

CLoraBuilder::CLoraBuilder( const CArray<CString>& _compositeClases )
{
	_compositeClases.CopyTo( compositeClasses );
}

void CLoraBuilder::BuildFcWrapper( CDnnLayerGraph& graph, const char* fcName, const CLoraParams& params ) const
{
	NeoAssert( graph.HasLayer( fcName ) );
	CPtr<CFullyConnectedLayer> fc = CheckCast<CFullyConnectedLayer>( graph.GetLayer( fcName ) );
	graph.DeleteLayer( *fc );

	NeoAssert( fc->Weights() != nullptr ); // "LoRA for uninitialized fully-connected" doesn't make sense 
	CPtr<CLoraFullyConnectedLayer> loraFc = FINE_DEBUG_NEW CLoraFullyConnectedLayer( *fc->Weights(),
		fc->FreeTerms(), params );
	loraFc->SetName( fc->GetName() );
	graph.AddLayer( *loraFc );
	loraFc->Connect( 0, fc->GetInputName( 0 ), fc->GetInputOutputNumber( 0 ) );
}

int CLoraBuilder::BuildAllFcWrappers( CDnnLayerGraph& rootGraph, const CLoraParams& params ) const
{
	auto impl = [this, &params] ( CDnnLayerGraph& currGraph, auto&& impl ) -> int
	{
		int result = 0;

		CArray<const char*> layerNames;
		currGraph.GetLayerList( layerNames );
		for( const char* layerName : layerNames ) {
			CPtr<CBaseLayer> layer = currGraph.GetLayer( layerName );

			CFullyConnectedLayer* fc = dynamic_cast<CFullyConnectedLayer*>( layer.Ptr() );
			if( fc != nullptr ) {
				BuildFcWrapper( currGraph, layerName, params );
				++result;
				continue;
			}

			const CString layerClass = GetLayerClass( *layer );
			if( compositeClasses.Find( layerClass ) != NotFound ) {
				CCompositeLayer* composite = dynamic_cast<CCompositeLayer*>( layer.Ptr() );
				NeoAssert( composite != nullptr );
				result += impl( *composite, impl ); // recurrent call of this lambda
			}
		}

		return result;
	};

	return impl( rootGraph, impl );
}

int CLoraBuilder::DisableNonLoraTraining( CDnnLayerGraph& graph ) const
{
	auto impl = [] ( CDnnLayerGraph& currGraph, auto&& impl ) -> int
	{
		int result = 0;

		CArray<const char*> layerNames;
		currGraph.GetLayerList( layerNames );
		for( const char* layerName : layerNames ) {
			CPtr<CBaseLayer> layer = currGraph.GetLayer( layerName );

			CLoraFullyConnectedLayer* loraFc = dynamic_cast<CLoraFullyConnectedLayer*>( layer.Ptr() );
			if( loraFc != nullptr ) {
				continue; // don't touch LoRA wrappers
			}

			CCompositeLayer* composite = dynamic_cast<CCompositeLayer*>( layer.Ptr() );
			if( composite != nullptr ) {
				result += impl( *composite, impl );
				continue;
			}

			if( layer->IsLearnable() && layer->IsLearningEnabled() ) {
				layer->DisableLearning();
				++result;
			}
		}

		return result;
	};

	return impl( graph, impl );
}

// Replaces specific fc wrapper with fc layer
// mergeWeights defines whether weights of the new fc will be original ones or emulates full LoRA wrapper
void CLoraBuilder::replaceFcWrapper( CDnnLayerGraph& graph, const char* fcName, bool mergeWeights ) const
{
	NeoAssert( graph.HasLayer( fcName ) );
	CPtr<CLoraFullyConnectedLayer> loraFc = CheckCast<CLoraFullyConnectedLayer>( graph.GetLayer( fcName ) );

	graph.DeleteLayer( *loraFc );

	CPtr<CFullyConnectedLayer> mergedFc = FINE_DEBUG_NEW CFullyConnectedLayer( loraFc->MathEngine(),
		loraFc->GetName() );
	mergedFc->SetNumberOfElements( loraFc->OutputSize() );
	mergedFc->Weights() = mergeWeights ? loraFc->GetMergedWeightsNoCopy() 
		: loraFc->GetSplitWeightsNoCopy();
	mergedFc->FreeTerms() = loraFc->GetFreeTermsNoCopy();
	mergedFc->Connect( 0, loraFc->GetInputName( 0 ), loraFc->GetInputOutputNumber( 0 ) );
	graph.AddLayer( *mergedFc );
}

int CLoraBuilder::replaceAllFcWrappers( CDnnLayerGraph& graph, bool mergeWeights ) const
{
	auto impl = [this, &mergeWeights] ( CDnnLayerGraph& currGraph, auto&& impl ) -> int
	{
		int result = 0;

		CArray<const char*> layerNames;
		currGraph.GetLayerList( layerNames );
		for( const char* layerName : layerNames ) {
			CPtr<CBaseLayer> layer = currGraph.GetLayer( layerName );

			CLoraFullyConnectedLayer* loraFc = dynamic_cast<CLoraFullyConnectedLayer*>( layer.Ptr() );
			if( loraFc != nullptr ) {
				replaceFcWrapper( currGraph, layerName, mergeWeights );
				++result;
				continue;
			}

			CCompositeLayer* composite = dynamic_cast<CCompositeLayer*>( layer.Ptr() );
			if( composite != nullptr ) {
				result += impl( *composite, impl );
			}
		}

		return result;
	};

	return impl( graph, impl );
}

//----------------------------------------------------------------------------------------------------------------------

namespace {

// Codes for signaling the type of wrapped layer
enum TLoraLayerType
{
	LLT_None, // special value used for signaling that there is no more layers
	LLT_FullyConnected,

	LLT_Count
};

} // anonymous namespace

static const int loraSerializerVersion = 0;

static int storeLora( CDnn& dnn, CArchive& archive )
{
	NeoAssert( archive.IsStoring() );
	( void ) archive.SerializeVersion( loraSerializerVersion );

	CArray<CString> layerPath;

	auto impl = [&archive, &layerPath] ( CDnnLayerGraph& graph, auto&& impl ) -> int
	{
		int result = 0;
		CArray<const char*> layerNames;
		graph.GetLayerList( layerNames );
		for( const char* layerName : layerNames ) {
			static_assert( LLT_Count == 2, "LLT_Count != 2" );

			CBaseLayer* layer = graph.GetLayer( layerName ).Ptr();

			// LoRA layers may inherit CCompositeLayer
			// That's why at first we check if current layer is a LoRA wrapper
			CLoraFullyConnectedLayer* loraFc = dynamic_cast<CLoraFullyConnectedLayer*>( layer );
			if( loraFc != nullptr ) {
				// For now only fully-connected is supported but in future embedding will probably be supported
				TLoraLayerType loraType = LLT_FullyConnected;
				archive.SerializeEnum( loraType );

				// Store full path to this layer
				// We don't use CBaseLayer::GetPath() here because we may get indistinguashable paths
				// when path separator is used inside of layer names (which sometimes the case esp for exported nets)
				// e.g. "ROOT" / "BLOCK0" / "FC0" vs. "ROOT/BLOCK0" / "FC0" both have path "ROOT/BLOCK0/FC0"
				layerPath.Add( loraFc->GetName() );
				archive.Serialize( layerPath );
				layerPath.DeleteLast();


				CLoraParams params( loraFc->Rank(), loraFc->Alpha(), loraFc->Dropout() );
				params.Serialize( archive );

				CPtr<CDnnBlob> aWeights = loraFc->GetAWeightsNoCopy();
				CPtr<CDnnBlob> bWeights = loraFc->GetBWeightsNoCopy();
				SerializeBlob( loraFc->MathEngine(), archive, aWeights );
				SerializeBlob( loraFc->MathEngine(), archive, bWeights );
				++result;
				continue;
			}

			CCompositeLayer* composite = dynamic_cast<CCompositeLayer*>( layer );
			if( composite != nullptr ) {
				// Updating current path and make recurrent call
				layerPath.Add( composite->GetName() );
				result += impl( *composite, impl );
				layerPath.DeleteLast();
			}
		}
		return result;
	};

	const int loraLayerCount = impl( dnn, impl ); // Store weights and calculate the amount of LoRA layers

	// Write special value to signal that there is no more LoRA weights in archive
	TLoraLayerType end = LLT_None;
	archive.SerializeEnum( end );

	return loraLayerCount;
}

static int loadLora( CDnn& dnn, CArchive& archive )
{
	CLoraBuilder loraBuilder;

	NeoAssert( archive.IsLoading() );
	( void ) archive.SerializeVersion( loraSerializerVersion );

	int loraLayerCount = 0;

	while( true ) {
		static_assert( LLT_Count == 2, "LLT_Count != 2" );

		TLoraLayerType layerType = LLT_None;
		archive.SerializeEnum( layerType );
		if( layerType == LLT_None ) {
			break;
		}

		// LoRA is supported only for fully-connected layers
		check( layerType == LLT_FullyConnected, ERR_BAD_ARCHIVE, archive.Name() );
		loraLayerCount++;

		CArray<CString> layerPath;
		archive.Serialize( layerPath );

		CLoraParams params;
		params.Serialize( archive );

		CDnnLayerGraph* currentGraph = &dnn;
		// Iterating from root dnn over all the composites
		for( int i = 0; i < layerPath.Size() - 1; ++i ) {
			CCompositeLayer* composite
				= dynamic_cast<CCompositeLayer*>( currentGraph->GetLayer( layerPath[i] ).Ptr() );
			// Intermediate layer in path must be composite
			check( composite != nullptr, ERR_BAD_ARCHIVE, archive.Name() );
			currentGraph = composite;
		}

		// path must contain at least name of last layer
		check( layerPath.Size() >= 1, ERR_BAD_ARCHIVE, archive.Name() );
		// replace fully-connected with lora wrapper if needed
		if( dynamic_cast<CFullyConnectedLayer*>( currentGraph->GetLayer( layerPath.Last() ).Ptr() ) != nullptr ) {
			loraBuilder.BuildFcWrapper( *currentGraph, layerPath.Last(), params );
		}

		CLoraFullyConnectedLayer* loraFc
			= dynamic_cast<CLoraFullyConnectedLayer*>( currentGraph->GetLayer( layerPath.Last() ).Ptr() );
		// mismatch between CDnn and lora weights (layer is missing)
		check( loraFc != nullptr, ERR_BAD_ARCHIVE, archive.Name() );

		CPtr<CDnnBlob> aWeights;
		SerializeBlob( dnn.GetMathEngine(), archive, aWeights );
		CPtr<CDnnBlob> bWeights;
		SerializeBlob( dnn.GetMathEngine(), archive, bWeights );

		loraFc->UpdateParams( params, aWeights.Ptr(), bWeights.Ptr() );
	}

	return loraLayerCount;
}

//----------------------------------------------------------------------------------------------------------------------

int CLoraSerializer::Serialize( CDnn& dnn, CArchive& archive ) const
{
	if( archive.IsStoring() ) {
		return storeLora( dnn, archive );
	} else if( archive.IsLoading() ) {
		return loadLora( dnn, archive );
	}

	NeoAssert( false );
	return 0;
}

int CLoraSerializer::Serialize( CDistributedTraining& distributed, CArchive& archive ) const
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

int CLoraSerializer::SerializeCheckpoint( CDnn& dnn, CArchive& archive ) const
{
	const int result = Serialize( dnn, archive ); // Serializing LoRA weights
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
