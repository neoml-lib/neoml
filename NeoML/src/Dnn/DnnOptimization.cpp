/* Copyright © 2017-2022 ABBYY Production LLC

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

#include <NeoML/Dnn/DnnOptimization.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/MobileNetV2BlockLayer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>

namespace NeoML {

namespace MobileNetV2 {

// Information about mobile net block detected in CDnn
struct CBlockInfo {
	CString InputName; // Name of the layer connected to ExpandConv (and Residual if present)
	int InputOutputIndex; // Output index if layer
	
	CConvLayer* ExpandConv; // Expand 1x1 convolution
	CReLULayer* ExpandReLU; // ReLU after Expand convolution
	CChannelwiseConvLayer* Channelwise; // Channelwise 3x3 convolution
	CReLULayer* ChannelwiseReLU; // ReLU after Channelwise convolution
	CConvLayer* DownConv; // Down 1x1 convolution
	CEltwiseSumLayer* Residual; // Residual (nullptr if block doesn't have residual connection)
};

// Marks layers from the block as deleted
// Allows to avoid adding one layer to 2 different blocks
static void markLayersAsDeleted( const CBlockInfo& info,
	CHashTable<CString>& layersToDelete )
{
	layersToDelete.Add( info.ExpandConv->GetName() );
	layersToDelete.Add( info.ExpandReLU->GetName() );
	layersToDelete.Add( info.Channelwise->GetName() );
	layersToDelete.Add( info.ChannelwiseReLU->GetName() );
	layersToDelete.Add( info.DownConv->GetName() );
	if( info.Residual != nullptr ) {
		layersToDelete.Add( info.Residual->GetName() );
	}
}

// Fills map with pairs:
//     layerName : number of inputs connected to its outputs
static void calculateOutputConnections( const CDnn& dnn, CMap<CString, int>& outputConnections )
{
	outputConnections.DeleteAll();
	CArray<const char*> layerList;
	dnn.GetLayerList( layerList );
	for( int i = 0; i < layerList.Size(); ++i ) {
		outputConnections.GetOrCreateValue( layerList[i] ) = 0;
	}
	for( int i = 0; i < layerList.Size(); ++i ) {
		const CBaseLayer* layer = dnn.GetLayer( layerList[i] );
		for( int inputIndex = 0; inputIndex < layer->GetInputCount(); ++inputIndex ) {
			outputConnections[layer->GetInputName( inputIndex )]++;
		}
	}
}

// Checks that there is a block ending with lastLayer
// If block is detected returns true and fills the info with details
// Otherwise returns false
static bool getMobileNetV2Block( const CMap<CString, int>& outputConnections, const CHashTable<CString>& layersToDelete,
	CBlockInfo& info, CDnn& dnn, CBaseLayer* lastLayer )
{
	CEltwiseSumLayer* residual = dynamic_cast<CEltwiseSumLayer*>( lastLayer );
	if( residual != nullptr ) {
		if( lastLayer->GetInputCount() != 2
			|| layersToDelete.Has( lastLayer->GetName() ) )
		{
			return false;
		}

		for( int input = 0; input < 2; ++input ) {
			// Try to interpret input as non-residual block and other as residual connection
			const int otherInput = 1 - input;
			if( getMobileNetV2Block( outputConnections, layersToDelete, info, dnn,
					dnn.GetLayer( lastLayer->GetInputName( input ) ).Ptr() )
				&& info.InputName == lastLayer->GetInputName( otherInput )
				&& info.InputOutputIndex == lastLayer->GetInputOutputNumber( otherInput )
				&& info.Residual == nullptr
				&& outputConnections[info.DownConv->GetName()] == 1 )
			{
				info.Residual = residual;
				return true;
			}
		}

		return false;
	}

	auto isValid1x1Conv = [&layersToDelete] ( const CConvLayer* conv ) -> bool
	{
		if( conv == nullptr
			|| layersToDelete.Has( conv->GetName() )
			|| conv->GetInputCount() > 1
			|| conv->GetFilterHeight() != 1 || conv->GetFilterWidth() != 1
			|| conv->GetPaddingHeight() != 0 || conv->GetPaddingWidth() != 0
			|| conv->GetStrideHeight() != 1 || conv->GetStrideWidth() != 1 )
		{
			return false;
		}
		return true;
	};

	auto isValidReLU = [&layersToDelete, &outputConnections] ( const CReLULayer* relu ) -> bool 
	{
		if( relu == nullptr
			|| layersToDelete.Has( relu->GetName() )
			|| relu->GetInputCount() > 1
			|| outputConnections[relu->GetName()] != 1 )
		{
			return false;
		}
		return true;
	};

	auto isValidChannelwiseConv = [&layersToDelete, &outputConnections] ( const CChannelwiseConvLayer* channelwise ) -> bool
	{
		if( channelwise == nullptr
			|| outputConnections[channelwise->GetName()] != 1
			|| layersToDelete.Has( channelwise->GetName() )
			|| channelwise->GetInputCount() > 1
			|| channelwise->GetInputCount() > 1
			|| channelwise->GetFilterHeight() != 3 || channelwise->GetFilterWidth() != 3
			|| channelwise->GetDilationHeight() != 1 || channelwise->GetDilationWidth() != 1
			|| channelwise->GetPaddingHeight() != 1 || channelwise->GetPaddingWidth() != 1
			|| ( channelwise->GetStrideHeight() != channelwise->GetStrideWidth() && channelwise->GetStrideHeight() > 2 ) )
		{
			return false;
		}
		return true;
	};

	// Try to find non-residual block
	info.Residual = nullptr;
	info.DownConv = dynamic_cast<CConvLayer*>( lastLayer );
	if( !isValid1x1Conv( info.DownConv ) ) {
		return false;
	}
	info.ChannelwiseReLU = dynamic_cast<CReLULayer*>( dnn.GetLayer( info.DownConv->GetInputName( 0 ) ).Ptr() );
	if( !isValidReLU( info.ChannelwiseReLU ) ) {
		return false;
	}
	info.Channelwise = dynamic_cast<CChannelwiseConvLayer*>( dnn.GetLayer( info.ChannelwiseReLU->GetInputName( 0 ) ).Ptr() );
	if( !isValidChannelwiseConv( info.Channelwise ) ) {
		return false;
	}
	info.ExpandReLU = dynamic_cast<CReLULayer*>( dnn.GetLayer( info.Channelwise->GetInputName( 0 ) ).Ptr() );
	if( !isValidReLU( info.ExpandReLU ) ) {
		return false;
	}
	info.ExpandConv = dynamic_cast<CConvLayer*>( dnn.GetLayer( info.ExpandReLU->GetInputName( 0 ) ).Ptr() );
	if( !isValid1x1Conv( info.ExpandConv ) || outputConnections[info.ExpandConv->GetName()] != 1 ) {
		return false;
	}
	info.InputName = info.ExpandConv->GetInputName( 0 );
	info.InputOutputIndex = info.ExpandConv->GetInputOutputNumber( 0 );
	return true;
}

static void replaceLayers( CDnn& dnn, const CArray<CBlockInfo>& blocksToReplace )
{
	int layersDeleted = 0;
	for( int blockIndex = 0; blockIndex < blocksToReplace.Size(); ++blockIndex ) {
		const CBlockInfo& info = blocksToReplace[blockIndex];
		CPtr<CMobileNetV2BlockLayer> mobileNetV2Block = new CMobileNetV2BlockLayer( dnn.GetMathEngine(),
			info.ExpandConv->GetFilterData(),
			!info.ExpandConv->IsZeroFreeTerm() ? info.ExpandConv->GetFreeTermData() : nullptr,
			info.ExpandReLU->GetUpperThreshold(), info.Channelwise->GetStrideWidth(), info.Channelwise->GetFilterData(),
			!info.Channelwise->IsZeroFreeTerm() ? info.Channelwise->GetFreeTermData() : nullptr,
			info.ChannelwiseReLU->GetUpperThreshold(), info.DownConv->GetFilterData(),
			!info.DownConv->IsZeroFreeTerm() ? info.DownConv->GetFreeTermData() : nullptr, info.Residual != nullptr );
		mobileNetV2Block->SetName( info.Residual != nullptr ? info.Residual->GetName() : info.DownConv->GetName() );
		dnn.DeleteLayer( *info.ExpandConv );
		dnn.DeleteLayer( *info.ExpandReLU );
		dnn.DeleteLayer( *info.Channelwise );
		dnn.DeleteLayer( *info.ChannelwiseReLU );
		dnn.DeleteLayer( *info.DownConv );
		layersDeleted += 5;
		if( info.Residual != nullptr ) {
			dnn.DeleteLayer( *info.Residual );
			layersDeleted++;
		}
		dnn.AddLayer( *mobileNetV2Block );
		mobileNetV2Block->Connect( 0, info.InputName, info.InputOutputIndex );
	}
}

static void optimizeDnn( CDnn& dnn, CDnnOptimizationReport& report )
{
	CArray<CBlockInfo> blocksToReplace;
	CHashTable<CString> layersToDelete;
	CMap<CString, int> outputConnections;
	CArray<const char*> layerList;

	// Step 1: replace residual blocks only
	calculateOutputConnections( dnn, outputConnections );
	dnn.GetLayerList( layerList );
	for( int i = 0; i < layerList.Size(); ++i ) {
		CEltwiseSumLayer* residual = dynamic_cast<CEltwiseSumLayer*>( dnn.GetLayer( layerList[i] ).Ptr() );
		CBlockInfo info;
		if( residual != nullptr
			&& getMobileNetV2Block( outputConnections, layersToDelete, info, dnn, residual ) )
		{
			markLayersAsDeleted( info, layersToDelete );
			blocksToReplace.Add( info );
		}
	}
	replaceLayers( dnn, blocksToReplace );
	report.MobileNetV2ResidualBlocks = blocksToReplace.Size();

	// Step 2: replace any blocks
	calculateOutputConnections( dnn, outputConnections );
	blocksToReplace.DeleteAll();
	layerList.DeleteAll();
	layersToDelete.DeleteAll();
	dnn.GetLayerList( layerList );
	for( int i = 0; i < layerList.Size(); ++i ) {
		CBlockInfo info;
		if( getMobileNetV2Block( outputConnections, layersToDelete, info, dnn, dnn.GetLayer( layerList[i] ).Ptr() ) ) {
			markLayersAsDeleted( info, layersToDelete );
			blocksToReplace.Add( info );
		}
	}
	replaceLayers( dnn, blocksToReplace );
	report.MobileNetV2NonResidualBlocks = blocksToReplace.Size();
}

} // namespace MobileNetV2

CDnnOptimizationReport OptimizeDnn( CDnn& dnn )
{
	CDnnOptimizationReport report;
	MobileNetV2::optimizeDnn( dnn, report );
	return report;
}

} // namespace NeoML
