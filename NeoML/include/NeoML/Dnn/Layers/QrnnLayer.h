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
// As a result recurrent part contains only eltwise operations.
// For optimization purposes this recurrent part is implemented without explicit RecurrentLayers.
//
// Layer inputs:
//     1. Blob with sequences BatchLength x BatchWidth x Height x Width x Depth x Channels where
//       - BatchLength - length of input sequences
//       - BatchWidth - number of sequences in the batch
//       - Height * Width * Depth * Channels - size of objects in sequences
//     2. Optional. Initial state for first recurrent part of size BatchWidth x Channels where
//       - BatchWidth must be equal to the BatchWidth of the first input
//       - Channels must be equal to the GetHiddenSize()
//     3. Optional. Initial state for the second recurrent part (bidirectional case only)
//       of the same size as previous layer input
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
	// Different poolings used in QRNN
	enum TPoolingType {
		PT_FPooling, // f-pooling from article, uses 2 gates (Update, Forget)
		PT_FoPooling, // fo-pooling from article, uses 3 gates (Update, Forget, Output)
		PT_IfoPooling, // ifo pooling from article, uses 4 gates (Update, Forget, Output, Input)

		PT_Count
	};

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

	// Layer settings

	// Changing next settings may cause weight reset
	// That's why they must be called before training

	// Pooling type used after the time convolution
	TPoolingType GetPoolingType() const { return poolingType; }
	void SetPoolingType( TPoolingType newPoolingType );

	// The way to process sequences in recurrent part
	TRecurrentMode GetRecurrentMode() const { return recurrentMode; }
	void SetRecurrentMode( TRecurrentMode newMode );

	// Activation function apllied to the update gate
	// Tanh by default
	TActivationFunction GetActivation() const { return activation; }
	void SetActivation( TActivationFunction newActivation );

	// Hidden state size
	int GetHiddenSize() const { return timeConv->GetFilterCount() / gateCount(); }
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

	// Next settings may be changed during training without losing trained weights

	// Dropout rate
	// By default is equal to zero (no dropout)
	float GetDropout() const { return dropout == nullptr ? 0.f : dropout->GetDropoutRate(); }
	void SetDropout( float rate );

	// Trainable parameters

	// Time convolution filter
	CPtr<CDnnBlob> GetFilterData() const { return timeConv->GetFilterData(); }
	void SetFilterData( const CPtr<CDnnBlob>& newFilter ) { timeConv->SetFilterData( newFilter ); }

	// Time convolution free terms
	CPtr<CDnnBlob> GetFreeTermData() const { return timeConv->GetFreeTermData(); }
	void SetFreeTermData( const CPtr<CDnnBlob>& newFreeTerm ) { timeConv->SetFreeTermData( newFreeTerm ); }

	void Serialize( CArchive& archive );

private:
	enum TGate {
		G_Update, // new activation values (Z in article)
		G_Forget, // forget multiplier (F in article)
		G_Output, // output gate, (O in article)
		G_Input, // input gate (I in article)

		G_Count
	};
	// Layer config
	TPoolingType poolingType; // pooling type used after time convolution
	TRecurrentMode recurrentMode; // method used for processing sequences
	TActivationFunction activation; // update gate activation function
	// Conv + split
	CPtr<CTimeConvLayer> timeConv; // time convolution
	CPtr<CSplitChannelsLayer> split; // split layer
	// Forget gate
	CPtr<CSigmoidLayer> forgetSigmoid; // forget gate sigmoid
	CPtr<CDropoutLayer> dropout; // forget gate dropout
	CPtr<CLinearLayer> postDropoutLinear; // compensate scale after dropout
	// Qrnn poolings
	CPtr<CBaseLayer> firstPooling;
	CPtr<CBaseLayer> secondPooling; // used when recurrent mode is bidirection

	void buildLayer( float dropoutRate, int hiddenSize, int windowSize, int stride, int padFront, int padBack );
	CPtr<CSigmoidLayer> addSigmoid( CBaseLayer& inputLayer, int outputIndex, const char* sigmoidName );
	CPtr<CBaseLayer> addPoolingLayer( const char* name, bool reverse );
	CPtr<CEltwiseMulLayer> addMulLayer( CBaseLayer& first, CBaseLayer& second, const char* mulLayerName );
	CPtr<CBaseLayer> addBidirectionalMerge( CBaseLayer& first, CBaseLayer& second, const char* mergeName );
	void addInitialStateInputMapping( CBaseLayer& pooling, int inputMappingIndex );
	void addDropout( float dropoutRate );
	void deleteDropout();
	void rebuildLayer( int prevGateCount );
	bool isBidirectional() const;
	int gateCount() const;
};

CLayerWrapper<CQrnnLayer> NEOML_API Qrnn( CQrnnLayer::TPoolingType poolingType, CQrnnLayer::TRecurrentMode recurrentMode,
	int hiddenSize, int windowSize, int paddingFront = 0, int paddingBack = 0, float dropout = 0.f,
	int stride = 1, TActivationFunction activation = AF_Tanh );

// --------------------------------------------------------------------------------------------------------------------

// Some layers that are used in QRNN

// f-pooling from QRNN
// the folrmula:
//
//  y_t = f_t * y_(t-1) + (1 - f_t) * z_t
//
// where t is index on temporal axis (along BatchLength)
// z - update gate
// f - forget gate
//
// Layer inputs:
//     1. update gate
//     2. forget gate
//     3. (optional) initial state
class NEOML_API CQrnnFPoolingLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CQrnnFPoolingLayer )
public:
	explicit CQrnnFPoolingLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CQrnnFPoolingLayer", false ) {}

	bool IsReverse() const { return reverse; }
	void SetReverse( bool newReverse ) { reverse = newReverse; }

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	bool reverse;
};

// if-pooling from QRNN
// the folrmula:
//
//  y_t = f_t * y_(t-1) + i_t * z_t
//
// where t is index on temporal axis (along BatchLength)
// z - update gate
// f - forget gate
// i - input gate
//
// Layer inputs:
//     1. update gate
//     2. forget gate
//     3. input gate
//     4. (optional) initial state

class NEOML_API CQrnnIfPoolingLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CQrnnIfPoolingLayer )
public:
	explicit CQrnnIfPoolingLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CQrnnIfPoolingLayer", false ) {}

	bool IsReverse() const { return reverse; }
	void SetReverse( bool newReverse ) { reverse = newReverse; }

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	bool reverse;
};

} // namespace NeoML
