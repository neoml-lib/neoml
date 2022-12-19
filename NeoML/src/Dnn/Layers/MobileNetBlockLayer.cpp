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

#include <NeoML/Dnn/Layers/MobileNetBlockLayer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>

namespace NeoML {

static const int CacheSize = 32 * 1024;

CMobileNetBlockLayer::CMobileNetBlockLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "MobileNetBlock", false ),
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

CMobileNetBlockLayer::~CMobileNetBlockLayer()
{
	if( convDesc != nullptr ) {
		delete convDesc;
	}
}

static const int MobileNetBlockLayerVersion = 0;

void CMobileNetBlockLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MobileNetBlockLayerVersion );
	CBaseLayer::Serialize( archive );
	archive.Serialize( residual );
}

void CMobileNetBlockLayer::Reshape()
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
	convDesc = MathEngine().InitBlobChannelwiseConvolution( channelwiseInputDesc, 1, 1, stride, stride, ChannelwiseFilter()->GetDesc(),
		ChannelwiseFreeTerm() != nullptr ? &ChannelwiseFreeTerm()->GetDesc() : nullptr, channelwiseOutputDesc );

	if( InputsMayBeOverwritten() && inputDescs[0].HasEqualDimensions( outputDescs[0] ) ) {
		NeoAssert( stride == 1 );
		EnableInPlace( true );
	}
}

void CMobileNetBlockLayer::RunOnce()
{
	MathEngine().RunMobileNetBlock( inputBlobs[0]->GetDesc(), outputBlobs[0]->GetDesc(), *convDesc,
		inputBlobs[0]->GetData(), ExpandFilter()->GetData(),
		ExpandFreeTerm() != nullptr ? &ExpandFreeTerm()->GetData<const float>() : nullptr,
		expandReLUThreshold, ChannelwiseFilter()->GetData(),
		ChannelwiseFreeTerm() != nullptr ? &ChannelwiseFreeTerm()->GetData<const float>() : nullptr,
		channelwiseReLUThreshold, DownFilter()->GetData(),
		DownFreeTerm() != nullptr ? &DownFreeTerm()->GetData<const float>() : nullptr, residual,
		outputBlobs[0]->GetData() );
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

static bool getMobileNetBlock( struct CBlockInfo& info, CDnn& dnn, CBaseLayer* lastLayer )
{
	CEltwiseSumLayer* residual = dynamic_cast<CEltwiseSumLayer*>( lastLayer );
	if( residual != nullptr ) {
		if( lastLayer->GetInputCount() != 2 ) {
			return false;
		}

		for( int input = 0; input < 2; ++input ) {
			const int otherInput = 1 - input;
			if( getMobileNetBlock( info, dnn, dnn.GetLayer( lastLayer->GetInputName( input ) ).Ptr() )
				&& info.InputName == lastLayer->GetInputName( otherInput )
				&& info.InputOutputIndex == lastLayer->GetInputOutputNumber( otherInput )
				&& info.Residual == nullptr )
			{
				info.Residual = residual;
				return true;
			}
		}

		return false;
	}

	auto isValid1x1Conv = [] ( const CConvLayer* conv ) -> bool
	{
		if( conv == nullptr ) {
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

		if( conv->GetDilationHeight() != 1 || conv->GetDilationWidth() != 1 ) {
			if( debugPrint ) ::printf( "1x1 conv with dilation: %s\n", conv->GetName() );
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

	auto isValidReLU = [] ( const CReLULayer* relu ) -> bool 
	{
		if( relu == nullptr ) {
			return false;
		}

		return true;
	};

	auto isValidChannelwiseConv = [] ( const CChannelwiseConvLayer* channelwise ) -> bool
	{
		if( channelwise == nullptr ) {
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
	if( !isValid1x1Conv( info.ExpandConv ) ) {
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
		CPtr<CMobileNetBlockLayer> mobileNetBlock = new CMobileNetBlockLayer( dnn.GetMathEngine() );
		mobileNetBlock->ExpandFilter() = info.ExpandConv->GetFilterData();
		mobileNetBlock->ExpandFreeTerm() = !info.ExpandConv->IsZeroFreeTerm() ? info.ExpandConv->GetFreeTermData() : nullptr;
		mobileNetBlock->ExpandReLUThreshold().SetValue( info.ExpandReLU->GetUpperThreshold() );
		mobileNetBlock->ChannelwiseFilter() = info.Channelwise->GetFilterData();
		mobileNetBlock->ChannelwiseFreeTerm() = !info.Channelwise->IsZeroFreeTerm() ? info.Channelwise->GetFreeTermData() : nullptr;
		mobileNetBlock->ChannelwiseReLUThreshold().SetValue( info.ChannelwiseReLU->GetUpperThreshold() );
		mobileNetBlock->DownFilter() = info.DownConv->GetFilterData();
		mobileNetBlock->DownFreeTerm() = !info.DownConv->IsZeroFreeTerm() ? info.DownConv->GetFreeTermData() : nullptr;
		mobileNetBlock->Stride() = info.Channelwise->GetStrideHeight();
		mobileNetBlock->Residual() = info.Residual != nullptr;
		mobileNetBlock->SetName( info.Residual != nullptr ? info.Residual->GetName() : info.DownConv->GetName() );
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
		dnn.AddLayer( *mobileNetBlock );
		mobileNetBlock->Connect( 0, info.InputName, info.InputOutputIndex );
	}

	if( debugPrint ) ::printf( "Replaced %d layers with %d blocks\n", layersDeleted, blocksToReplace.Size() );
}

int ReplaceMobileNetBlocks( CDnn& dnn )
{
	// Step 1: replacing residual blocks because non-residual blocks are part of residual
	CArray<CBlockInfo> blocksToReplace;
	CArray<const char*> layerList;
	dnn.GetLayerList( layerList );
	for( int i = 0; i < layerList.Size(); ++i ) {
		CEltwiseSumLayer* residual = dynamic_cast<CEltwiseSumLayer*>( dnn.GetLayer( layerList[i] ).Ptr() );
		CBlockInfo info;
		if( residual != nullptr && getMobileNetBlock( info, dnn, residual ) ) {
			blocksToReplace.Add( info );
		}
	}
	replaceLayers( dnn, blocksToReplace );
	const int residualBlocksReplaced = blocksToReplace.Size();
	blocksToReplace.DeleteAll();

	layerList.DeleteAll();
	dnn.GetLayerList( layerList );
	for( int i = 0; i < layerList.Size(); ++i ) {
		CBlockInfo info;
		if( getMobileNetBlock( info, dnn, dnn.GetLayer( layerList[i] ).Ptr() ) ) {
			blocksToReplace.Add( info );
		}
	}
	replaceLayers( dnn, blocksToReplace );

	debugPrint = false;
	return residualBlocksReplaced + blocksToReplace.Size();
}

} // namespace NeoML
