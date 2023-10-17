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

void CLoraBuilder::BuildForFc( CDnnLayerGraph& graph, const char* fcName, const CLoraParams& params ) const
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

int CLoraBuilder::BuildForAllFcs( CDnnLayerGraph& rootGraph, const CLoraParams& params ) const
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
				BuildForFc( currGraph, layerName, params );
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

//----------------------------------------------------------------------------------------------------------------------

static const int loraSerializerVersion = 0;

static int storeLora( CDnn& dnn, CArchive& archive )
{
	NeoAssert( archive.IsStoring() );
	( void ) archive.SerializeVersion( loraSerializerVersion );

	// Save current file offset in order to overwrite the value afterwards
	const __int64 loraLayerCountOffset = archive.GetPosition();
	int loraLayerCount = 0;
	archive.Serialize( loraLayerCount );

	CArray<CString> layerPath;

	auto impl = [&archive, &layerPath] ( CDnnLayerGraph& graph, auto&& impl ) -> int
	{
		int result = 0;
		CArray<const char*> layerNames;
		graph.GetLayerList( layerNames );
		for( const char* layerName : layerNames ) {
			CBaseLayer* layer = graph.GetLayer( layerName ).Ptr();

			// LoRA layers may inherit CCompositeLayer
			// That's why at first we check if current layer is a LoRA wrapper
			CLoraFullyConnectedLayer* loraFc = dynamic_cast< CLoraFullyConnectedLayer* >( layer );
			if( loraFc != nullptr ) {
				// Store full path to this layer
				// We don't use CBaseLayer::GetPath() here because we may get indistinguashable paths
				// when path separator is used inside of layer names (which sometimes the case esp for exported nets)
				// e.g. "ROOT" / "BLOCK0" / "FC0" vs. "ROOT/BLOCK0" / "FC0" both have path "ROOT/BLOCK0/FC0"
				layerPath.Add( loraFc->GetName() );
				archive.Serialize( layerPath );
				layerPath.DeleteLast();

				// For now only fully-connected is supported but in future embedding will probably be supported
				int loraType = 0;
				archive.Serialize( loraType );

				CLoraParams params( loraFc->Rank(), loraFc->Alpha(), loraFc->Dropout() );
				params.Serialize( archive );

				CPtr<CDnnBlob> aWeights = loraFc->GetAWeightsNoCopy();
				CPtr<CDnnBlob> bWeights = loraFc->GetBWeightsNoCopy();
				SerializeBlob( loraFc->MathEngine(), archive, aWeights );
				SerializeBlob( loraFc->MathEngine(), archive, bWeights );
				++result;
				continue;
			}

			CCompositeLayer* composite = dynamic_cast< CCompositeLayer* >( layer );
			if( composite != nullptr ) {
				// Updating current path and make recurrent call
				layerPath.Add( composite->GetName() );
				result += impl( *composite, impl );
				layerPath.DeleteLast();
			}
		}
		return result;
	};

	loraLayerCount = impl( dnn, impl ); // Store weights and calculate the amount of LoRA layers

	// Write actual layer count over previously stored offset
	const __int64 loraEndOffset = archive.GetPosition();
	archive.Seek( loraLayerCountOffset, CBaseFile::begin );
	archive.Serialize( loraLayerCount );
	archive.Seek( loraEndOffset, CBaseFile::begin ); // don't forget to move back to the end of LoRA weightss

	return loraLayerCount;
}

static int loadLora( CDnn& dnn, CArchive& archive )
{
	NeoAssert( archive.IsLoading() );
	( void ) archive.SerializeVersion( loraSerializerVersion );

	int loraLayerCount = 0;
	archive.Serialize( loraLayerCount );

	for( int loraLayerIndex = 0; loraLayerIndex < loraLayerCount; ++loraLayerIndex ) {
		CArray<CString> layerPath;
		archive.Serialize( layerPath );

		int layerType = 0;
		archive.Serialize( layerType );
		check( layerType == 0, ERR_BAD_ARCHIVE, archive.Name() ); // LoRA is supported only for fully-connected layers

		CLoraParams params;
		params.Serialize( archive );

		CDnnLayerGraph* currentGraph = &dnn;
		// Iterating from root dnn over all the composites
		for( int i = 0; i < layerPath.Size() - 1; ++i ) {
			CCompositeLayer* composite = dynamic_cast< CCompositeLayer* >( currentGraph->GetLayer( layerPath[i] ).Ptr() );
			check( composite != nullptr, ERR_BAD_ARCHIVE, archive.Name() ); // Intermediate layer in path must be composite
			currentGraph = composite;
		}

		check( layerPath.Size() >= 1, ERR_BAD_ARCHIVE, archive.Name() ); // path must contain at least name of last layer
		// replace fully-connected with lora wrapper if needed
		if( dynamic_cast<CFullyConnectedLayer*>( currentGraph->GetLayer( layerPath.Last() ).Ptr() ) != nullptr ) {
			CLoraBuilder().BuildForFc( *currentGraph, layerPath.Last(), params );
		}

		CLoraFullyConnectedLayer* loraFc = dynamic_cast<CLoraFullyConnectedLayer*>( currentGraph->GetLayer( layerPath.Last() ).Ptr() );
		check( loraFc != nullptr, ERR_BAD_ARCHIVE, archive.Name() ); // mismatch between CDnn and lora weights (layer is missing)

		CPtr<CDnnBlob> aWeights;
		SerializeBlob( dnn.GetMathEngine(), archive, aWeights );
		CPtr<CDnnBlob> bWeights;
		SerializeBlob( dnn.GetMathEngine(), archive, bWeights );

		loraFc->UpdateParams( params, aWeights.Ptr(), bWeights.Ptr() );
	}

	return loraLayerCount;
}

//----------------------------------------------------------------------------------------------------------------------

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

int CLoraSerializer::SerializeCheckpoint( CDnn& dnn, CArchive& archive )
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

//----------------------------------------------------------------------------------------------------------------------

//void CLoraMerger::MergeFc( CDnnLayerGraph& graph, const char* fcName )
//{
//	NeoAssert( graph.HasLayer( fcName ) );
//	CPtr<CLoraFullyConnectedLayer> loraFc = CheckCast<CLoraFullyConnectedLayer>( graph.GetLayer( fcName ) );
//
//	graph.DeleteLayer( *loraFc );
//
//	CPtr<CFullyConnectedLayer> mergedFc = FINE_DEBUG_NEW CFullyConnectedLayer( loraFc->MathEngine(),
//		loraFc->GetName() );
//	mergedFc->SetNumberOfElements( loraFc->OutputSize() );
//	mergedFc->Weights() = loraFc->GetMergedBaseWeightsNoCopy();
//	mergedFc->FreeTerms() = loraFc->GetFreeTermsNoCopy();
//	mergedFc->Connect( 0, loraFc->GetInputName( 0 ), loraFc->GetInputOutputNumber( 0 ) );
//	graph.AddLayer( *mergedFc );
//
//	loraFc.Release();
//}

} // namespace NeoML
