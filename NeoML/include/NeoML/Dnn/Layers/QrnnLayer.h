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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/RecurrentLayer.h>
#include <NeoML/Dnn/Layers/TimeConvLayer.h>
#include <NeoML/Dnn/Layers/SplitLayer.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>

namespace NeoML {

// Quasi-recurrent layer
//
// The key idea is to reduce amount of calculations made in recurrent part
// in order to increase performance on GPU
//
// It's achieved by using 1-dimensional convolution over temporal axis (CTimeConvLayer) outside of recurrent part
// instead of FullyConnected layers inside of recurrent part.
//
// As a result recurrent part is reduced down to 2 eltwise layers.
//
// Layer inputs:
//     1. Blob with sequences BatchLength x BatchWidth x Height x Width x Depth x Channels where
//       - BatchLength - length of input sequences
//       - BatchWidth - number of sequences in the batch
//       - Height * Width * Depth * Channels - size of objects in sequences
//     2. Optional. Initial state for recurrent part with size BatchWidth x Channels where
//       - BatchWidth must be equal to the BatchWidth of the first input
//       - Channels must be equal to the GetHiddenSize()
//
// Layer outputs:
//     1. Results. It has a shape of BatchLength x BatchWidth x Channels where
//       - BatchLength is equal to (InputBatchLength + Padding - WindowSize + 1) / Stride + 1
//       - BatchWidth is equal to InputBatchWidth
//       - Channels is equal to 2 * GetHiddenSize() if recurrrent mode is RM_BidirectionalConcat.
//           Otherwise it's equal to GetHiddenSize()
//
// article: https://arxiv.org/pdf/1611.01576.pdf
class NEOML_API CQrnnLayer: public CCompositeLayer {
	NEOML_DNN_LAYER( CQrnnLayer )
public:
	// Different approaches in sequence processing
	enum TRecurrentMode {
		RM_Direct,
		RM_Reverse,

		// Bidirectional mode where two recurrent parts share the same time convolution
		RM_BidirectionalConcat, // returns the concatenation of direct and reverse recurrents
		RM_BidirectionalSum, // returns the sum of direct and reverse recurrents
		// If you want to use bidirectional qrnn with two separate time convolutions create 2 CQrnnLayers
		// and merge the results by CConcatChannelsLayer or CEltwiseSumLayer

		RM_Count
	};

	explicit CQrnnLayer( IMathEngine& mathEngine );

	// Hidden state size
	int GetHiddenSize() const { return timeConv->GetFilterCount() / 3; }
	void SetHiddenSize( int hiddenSize );

	// Window size
	// 1 by default
	int GetWindowSize() const { return timeConv->GetFilterSize(); }
	void SetWindowSize( int windowSize );

	// Window stride
	// 1 by default
	int GetStride() const { return timeConv->GetStride(); }
	void SetStride( int stride );

	// Padding
	// Adds zeros to the beginnings of the sequences
	int GetPaddingFront() const { return timeConv->GetPaddingFront(); }
	void SetPaddingFront( int padding );
	// Adds zeros to the ends of the sequences
	int GetPaddingBack() const { return timeConv->GetPaddingBack(); }
	void SetPaddingBack( int padding );
	
	// Activation function apllied to the update gate
	// Tanh by default
	TActivationFunction GetActivation() const { return activation; }
	void SetActivation( TActivationFunction newActivation );

	// Dropout rate
	// By default is equal to zero (no dropout)
	float GetDropout() const { return dropout == nullptr ? 0.f : dropout->GetDropoutRate(); }
	void SetDropout( float rate );

	// Time convolution filter
	CPtr<CDnnBlob> GetFilterData() const { return timeConv->GetFilterData(); }
	void SetFilterData( const CPtr<CDnnBlob>& newFilter ) { timeConv->SetFilterData( newFilter ); }

	// Time convolution free terms
	CPtr<CDnnBlob> GetFreeTermData() const { return timeConv->GetFreeTermData(); }
	void SetFreeTermData( const CPtr<CDnnBlob>& newFreeTerm ) { timeConv->SetFreeTermData( newFreeTerm ); }

	// The way to process sequences in recurrent part
	TRecurrentMode GetRecurrentMode() const { return recurrentMode; }
	void SetRecurrentMode( TRecurrentMode newMode );

	void Serialize( CArchive& archive );

private:
	enum TGate {
		G_Update, // new activation values (Z in article)
		G_Forget, // forget multiplier (F in article)
		G_Output, // output gate, (O in article)

		G_Count
	};
	TActivationFunction activation; // update gate activation function
	TRecurrentMode recurrentMode;
	CPtr<CTimeConvLayer> timeConv; // time convolution
	CPtr<CSplitChannelsLayer> split; // split layer
	CPtr<CSigmoidLayer> forgetSigmoid; // forget gate sigmoid
	CPtr<CDropoutLayer> dropout; // forget gate dropout (if needed)
	CPtr<CLinearLayer> postDropoutLinear; // compensate scale after dropout (if needed)
	CPtr<CEltwiseNegMulLayer> negForgetByUpdateByOutput; // (1 - forget) * update * output
	CPtr<CEltwiseMulLayer> forgetByOutput; // forget * output
	CPtr<CRecurrentLayer> firstRecurrent; // first recurrent part
	CPtr<CBackLinkLayer> firstBackLink; // back link from the first recurrent part
	CPtr<CRecurrentLayer> secondRecurrent; // additional recurrent part used in bidirectional case
	CPtr<CBackLinkLayer> secondBackLink; // back link from the second recurrent part
	CPtr<CBaseLayer> bidirectionalMerge; // merge layer for direct and reverse recurrent parts

	void buildLayer();
	CPtr<CRecurrentLayer> buildRecurrentPart( const char* name );
	void addDropout( float dropoutRate );
	void deleteDropout();
	void createBidirectionalLayers();
	void deleteBidirectionalLayers();
};

} // namespace NeoML
