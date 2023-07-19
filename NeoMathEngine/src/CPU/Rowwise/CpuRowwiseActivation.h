/* Copyright © 2017-2023 ABBYY

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

#include <CpuMathEnginePrivate.h>
#include "CpuRowwiseInterface.h"

namespace NeoML {

class CRowwiseActivation : public IRowwiseCpuImpl, public CRowwiseOperationDesc {
public:
	explicit CRowwiseActivation( const CActivationDesc& desc );

	TActivationFunction Type() const { return desc.GetType(); }

	// IRowwiseCpuImpl
	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int InputRowRequirement() const override { return 0; }
	int OutputRowRequirement() const override { return 0; }
	int InOperationBufferSize() const override { return 0; }
	int OutputRowCount() const override { return rowCount; }
	int OutputRowSize() const override { return rowSize; }
	bool IsTrivial() const override { return true; }
	CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const override;

private:
	CActivationDesc desc;
	int rowCount;
	int rowSize;
};

//---------------------------------------------------------------------------------------------------------------------

inline CRowwiseActivation::CRowwiseActivation( const CActivationDesc& desc ) :
	desc( desc ),
	rowCount( 0 ),
	rowSize( 0 )
{
}

inline CBlobDesc CRowwiseActivation::Reshape( const CBlobDesc& inputSize )
{
	rowCount = inputSize.ObjectCount() * inputSize.Height();
	rowSize = inputSize.Width() * inputSize.Channels();
	return inputSize;
}

inline IRowwiseCpuImpl::CProcessingReport CRowwiseActivation::Process( const float* input, int inputRowIndex,
	int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable, float* ) const
{
	CProcessingReport result;
	result.OutputRowsCalculated = std::min( outputRowsAvailable, inputRowIndex + inputRowsAvailable - outputRowIndex );
	result.InputRowsMayBeRemoved = outputRowIndex + result.OutputRowsCalculated - inputRowIndex;

	if( inputRowIndex < outputRowIndex ) {
		input += ( outputRowIndex - inputRowIndex ) * rowSize;
	}

	const int dataSize = result.OutputRowsCalculated * rowSize;
	switch( desc.GetType() ) {
		case AF_ELU:
			vectorELU( input, output, desc.GetParam<CELUActivationParam>().Alpha, dataSize );
			break;
		case AF_HardSigmoid:
			vectorHardSigmoid( input, output, desc.GetParam<CHardSigmoidActivationParam>().Slope,
				desc.GetParam<CHardSigmoidActivationParam>().Bias, dataSize );
			break;
		case AF_HardTanh:
			vectorMinMax( input, output, -1.f, 1.f, dataSize );
			break;
		case AF_HSwish:
			vectorHSwish( input, output, dataSize );
			break;
		case AF_LeakyReLU:
			vectorLeakyReLU( input, output, desc.GetParam<CLeakyReLUActivationParam>().Alpha, dataSize );
			break;
		case AF_Linear:
			if( desc.GetParam<CLinearActivationParam>().Multiplier != 1.f ) {
				vectorMultiply( input, output, desc.GetParam<CLinearActivationParam>().Multiplier, dataSize );
				input = output;
			}
			if( desc.GetParam<CLinearActivationParam>().FreeTerm != 0.f ) {
				vectorAddValue( input, output, dataSize, desc.GetParam<CLinearActivationParam>().FreeTerm );
				input = output;
			}
			if( input != output ) {
				// Corner case: Linear( 1, 0 ), not in-place
				dataCopy( output, input, dataSize );
			}
			break;
		case AF_ReLU:
			if( desc.GetParam<CReLUActivationParam>().UpperThreshold <= 0 ) {
				vectorReLU( input, output, dataSize );
			} else {
				vectorReLU( input, output, dataSize, desc.GetParam<CReLUActivationParam>().UpperThreshold );
			}
			break;
		case AF_Sigmoid:
			vectorSigmoid( input, output, dataSize );
			break;
		case AF_Tanh:
			vectorTanh( input, output, dataSize );
			break;
		default:
			ASSERT_EXPR( false );
	}

	return result;
}

} // namespace NeoML
