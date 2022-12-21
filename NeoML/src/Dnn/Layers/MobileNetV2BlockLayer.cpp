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

#include <NeoML/Dnn/Layers/MobileNetV2BlockLayer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>

namespace NeoML {

static const int CacheSize = 32 * 1024;

CMobileNetV2BlockLayer::CMobileNetV2BlockLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "MobileNetV2Block", false ),
	residual( false ),
	stride( 0 ),
	convDesc( nullptr ),
	expandReLUThreshold( mathEngine, 1 ),
	channelwiseReLUThreshold( mathEngine, 1 )
{
	expandReLUThreshold.SetValue( -1.f );
	channelwiseReLUThreshold.SetValue( -1.f );
	paramBlobs.SetSize( P_Count );
}

CMobileNetV2BlockLayer::~CMobileNetV2BlockLayer()
{
	if( convDesc != nullptr ) {
		delete convDesc;
	}
}

static const int MobileNetV2BlockLayerVersion = 0;

void CMobileNetV2BlockLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MobileNetV2BlockLayerVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( residual );
	archive.Serialize( stride );

	float expandReLU = 0;
	float channelwiseReLU = 0;
	if( archive.IsStoring() ) {
		expandReLU = expandReLUThreshold.GetValue();
		channelwiseReLU = channelwiseReLUThreshold.GetValue();
	}
	archive.Serialize( expandReLU );
	archive.Serialize( channelwiseReLU );
	if( archive.IsLoading() ) {
		expandReLUThreshold.SetValue( expandReLU );
		channelwiseReLUThreshold.SetValue( channelwiseReLU );
	}
}

void CMobileNetV2BlockLayer::Reshape()
{
	CheckInput1();

	NeoAssert( inputDescs[0].Depth() == 1 );
	const int inputChannels = inputDescs[0].Channels();

	NeoAssert( paramBlobs[P_ExpandFilter] != nullptr );
	const int expandedChannels = paramBlobs[P_ExpandFilter]->GetObjectCount();
	NeoAssert( paramBlobs[P_ExpandFilter]->GetHeight() == 1 );
	NeoAssert( paramBlobs[P_ExpandFilter]->GetWidth() == 1 );
	NeoAssert( paramBlobs[P_ExpandFilter]->GetDepth() == 1 );
	NeoAssert( paramBlobs[P_ExpandFilter]->GetChannelsCount() == inputChannels );
	if( paramBlobs[P_ExpandFreeTerm] != nullptr ) {
		NeoAssert( paramBlobs[P_ExpandFreeTerm]->GetDataSize() == expandedChannels );
	}

	NeoAssert( stride == 1 || stride == 2 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter] != nullptr );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetObjectCount() == 1 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetHeight() == 3 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetWidth() == 3 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetDepth() == 1 );
	NeoAssert( paramBlobs[P_ChannelwiseFilter]->GetChannelsCount() == expandedChannels );
	if( paramBlobs[P_ChannelwiseFreeTerm] != nullptr ) {
		NeoAssert( paramBlobs[P_ChannelwiseFreeTerm]->GetDataSize() == expandedChannels );
	}

	NeoAssert( paramBlobs[P_DownFilter] != nullptr );
	const int outputChannels = paramBlobs[P_DownFilter]->GetObjectCount();
	NeoAssert( paramBlobs[P_DownFilter]->GetHeight() == 1 );
	NeoAssert( paramBlobs[P_DownFilter]->GetWidth() == 1 );
	NeoAssert( paramBlobs[P_DownFilter]->GetDepth() == 1 );
	NeoAssert( paramBlobs[P_DownFilter]->GetChannelsCount() == expandedChannels );
	if( paramBlobs[P_DownFreeTerm] != nullptr ) {
		NeoAssert( paramBlobs[P_DownFreeTerm]->GetDataSize() == outputChannels );
	}

	NeoAssert( !residual || ( inputChannels == outputChannels && stride == 1 ) );

	outputDescs[0] = inputDescs[0];
	if( stride == 2 ) {
		outputDescs[0].SetDimSize( BD_Height, ( inputDescs[0].Height() + 1 ) / 2 );
		outputDescs[0].SetDimSize( BD_Width, ( inputDescs[0].Width() + 1 ) / 2 );
	}
	outputDescs[0].SetDimSize( BD_Channels, outputChannels );

	if( convDesc != nullptr ) {
		delete convDesc;
		convDesc = nullptr;
	}
	channelwiseInputDesc = inputDescs[0];
	channelwiseInputDesc.SetDimSize( BD_Channels, expandedChannels );
	channelwiseOutputDesc = outputDescs[0];
	channelwiseOutputDesc.SetDimSize( BD_Channels, expandedChannels );
	CBlobDesc freeTermDesc = paramBlobs[P_ChannelwiseFreeTerm] != nullptr
		? paramBlobs[P_ChannelwiseFreeTerm]->GetDesc() : CBlobDesc();
	convDesc = MathEngine().InitBlobChannelwiseConvolution( channelwiseInputDesc, 1, 1, stride, stride,
		paramBlobs[P_ChannelwiseFilter]->GetDesc(),
		paramBlobs[P_ChannelwiseFreeTerm] != nullptr ? &freeTermDesc : nullptr, channelwiseOutputDesc );

	if( InputsMayBeOverwritten() && inputDescs[0].HasEqualDimensions( outputDescs[0] ) ) {
		NeoAssert( stride == 1 );
		EnableInPlace( true );
	}
}

void CMobileNetV2BlockLayer::RunOnce()
{
	const CConstFloatHandle expandFt = paramBlobs[P_ExpandFreeTerm] == nullptr ? CConstFloatHandle()
		: paramBlobs[P_ExpandFreeTerm]->GetData<const float>();
	const CConstFloatHandle channelwiseFt = paramBlobs[P_ChannelwiseFreeTerm] == nullptr ? CConstFloatHandle()
		: paramBlobs[P_ChannelwiseFreeTerm]->GetData<const float>();
	const CConstFloatHandle downFt = paramBlobs[P_DownFreeTerm] == nullptr ? CConstFloatHandle()
		: paramBlobs[P_DownFreeTerm]->GetData<const float>();
	MathEngine().MobileNetV2Block( inputBlobs[0]->GetDesc(), outputBlobs[0]->GetDesc(), *convDesc,
		inputBlobs[0]->GetData(), paramBlobs[P_ExpandFilter]->GetData(), expandFt.IsNull() ? nullptr : &expandFt,
		expandReLUThreshold, paramBlobs[P_ChannelwiseFilter]->GetData(),
		channelwiseFt.IsNull() ? nullptr : &channelwiseFt, channelwiseReLUThreshold, 
		paramBlobs[P_DownFilter]->GetData(), downFt.IsNull() ? nullptr : &downFt, residual,
		outputBlobs[0]->GetData() );
}

CPtr<CDnnBlob> CMobileNetV2BlockLayer::getParamBlob( TParam param ) const
{
	if( paramBlobs[param] == nullptr ) {
		return nullptr;
	}

	return paramBlobs[param]->GetCopy();
}

void CMobileNetV2BlockLayer::setParamBlob( TParam param, const CPtr<CDnnBlob>& blob )
{
	paramBlobs[param] = blob == nullptr ? nullptr : blob->GetCopy();
}

// --------------------------------------------------------------------------------------------------------------------

static bool debugPrint = true;

struct CBlockInfo {
	CString InputName;
	int InputOutputIndex;
	
	CConvLayer* ExpandConv;
	CReLULayer* ExpandReLU;
	CChannelwiseConvLayer* Channelwise;
	CReLULayer* ChannelwiseReLU;
	CConvLayer* DownConv;
	CEltwiseSumLayer* Residual;
};

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

static bool getMobileNetV2Block( const CMap<CString, int>& outputConnections, const CHashTable<CString>& layersToDelete,
	CBlockInfo& info, CDnn& dnn, CBaseLayer* lastLayer )
{
	CEltwiseSumLayer* residual = dynamic_cast<CEltwiseSumLayer*>( lastLayer );
	if( residual != nullptr ) {
		if( lastLayer->GetInputCount() != 2 ) {
			return false;
		}

		if( layersToDelete.Has( lastLayer->GetName() ) ) {
			return false;
		}

		for( int input = 0; input < 2; ++input ) {
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
		if( conv == nullptr ) {
			return false;
		}

		if( layersToDelete.Has( conv->GetName() ) ) {
			return false;
		}

		if( conv->GetInputCount() > 1 ) {
			if( debugPrint ) ::printf( "conv with multiple inputs: %s\n", conv->GetName() );
			return false;
		}

		if( conv->GetFilterHeight() != 1 || conv->GetFilterWidth() != 1 ) {
			if( debugPrint ) ::printf( "not 1x1 conv: %s\n", conv->GetName() );
			return false;
		}

		if( conv->GetPaddingHeight() != 0 || conv->GetPaddingWidth() != 0 ) {
			if( debugPrint ) ::printf( "1x1 conv with padding: %s\n", conv->GetName() );
			return false;
		}

		if( conv->GetStrideHeight() != 1 || conv->GetStrideWidth() != 1 ) {
			if( debugPrint ) ::printf( "1x1 conv with strides: %s\n", conv->GetName() );
			return false;
		}

		return true;
	};

	auto isValidReLU = [&layersToDelete, &outputConnections] ( const CReLULayer* relu ) -> bool 
	{
		if( relu == nullptr ) {
			return false;
		}

		if( layersToDelete.Has( relu->GetName() ) ) {
			return false;
		}

		if( relu->GetInputCount() > 1 ) {
			if( debugPrint ) ::printf( "relu with multiple inputs: %s\n", relu->GetName() );
			return false;
		}

		if( outputConnections[relu->GetName()] != 1 ) {
			return false;
		}

		return true;
	};

	auto isValidChannelwiseConv = [&layersToDelete, &outputConnections] ( const CChannelwiseConvLayer* channelwise ) -> bool
	{
		if( channelwise == nullptr ) {
			return false;
		}

		if( outputConnections[channelwise->GetName()] != 1 ) {
			return false;
		}

		if( layersToDelete.Has( channelwise->GetName() ) ) {
			return false;
		}

		if( channelwise->GetInputCount() > 1 ) {
			if( debugPrint ) ::printf( "channelwise with multiple inputs: %s\n", channelwise->GetName() );
			return false;
		}

		if( channelwise->GetInputCount() > 1 ) {
			if( debugPrint ) ::printf( "conv with multiple inputs: %s\n", channelwise->GetName() );
			return false;
		}

		if( channelwise->GetFilterHeight() != 3 || channelwise->GetFilterWidth() != 3 ) {
			if( debugPrint ) ::printf( "channelwise with wrong filter size: %s\n", channelwise->GetName() );
			return false;
		}

		if( channelwise->GetDilationHeight() != 1 || channelwise->GetDilationWidth() != 1 ) {
			if( debugPrint ) ::printf( "channelwise with dilation: %s\n", channelwise->GetName() );
			return false;
		}

		if( channelwise->GetPaddingHeight() != 1 || channelwise->GetPaddingWidth() != 1 ) {
			if( debugPrint ) ::printf( "channelwise with wrong padding: %s\n", channelwise->GetName() );
			return false;
		}

		if( channelwise->GetStrideHeight() != channelwise->GetStrideWidth()
			&& channelwise->GetStrideHeight() > 2 )
		{
			if( debugPrint ) ::printf( "channelwise with wrong strides: %s\n", channelwise->GetName() );
			return false;
		}

		return true;
	};

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
		CPtr<CMobileNetV2BlockLayer> mobileNetV2Block = new CMobileNetV2BlockLayer( dnn.GetMathEngine() );
		mobileNetV2Block->SetExpandFilter( info.ExpandConv->GetFilterData() );
		mobileNetV2Block->SetExpandFreeTerm( !info.ExpandConv->IsZeroFreeTerm() ? info.ExpandConv->GetFreeTermData() : nullptr );
		mobileNetV2Block->SetExpandReLUThreshold( info.ExpandReLU->GetUpperThreshold() );
		mobileNetV2Block->SetChannelwiseFilter( info.Channelwise->GetFilterData() );
		mobileNetV2Block->SetChannelwiseFreeTerm( !info.Channelwise->IsZeroFreeTerm() ? info.Channelwise->GetFreeTermData() : nullptr );
		mobileNetV2Block->SetChannelwiseReLUThreshold( info.ChannelwiseReLU->GetUpperThreshold() );
		mobileNetV2Block->SetDownFilter( info.DownConv->GetFilterData() );
		mobileNetV2Block->SetDownFreeTerm( !info.DownConv->IsZeroFreeTerm() ? info.DownConv->GetFreeTermData() : nullptr );
		mobileNetV2Block->SetStride( info.Channelwise->GetStrideHeight() );
		mobileNetV2Block->SetResidual( info.Residual != nullptr );
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

	if( debugPrint ) ::printf( "Replaced %d layers with %d blocks\n", layersDeleted, blocksToReplace.Size() );
}

int ReplaceMobileNetV2Blocks( CDnn& dnn )
{
	CArray<CBlockInfo> blocksToReplace;
	CHashTable<CString> layersToDelete;
	CMap<CString, int> outputConnections;
	CArray<const char*> layerList;

	// Step 1: replace residual blocks only because non-residual blocks are part of residual
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
	const int residualBlocksReplaced = blocksToReplace.Size();

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

	debugPrint = false;
	return residualBlocksReplaced + blocksToReplace.Size();
}

} // namespace NeoML
