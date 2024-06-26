/* Copyright © 2023-2024 ABBYY

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

#include "CudaRowwiseInterface.h"
#include "../CudaMathEngine.h"

namespace NeoML {

class CCudaRowwiseActivation : public ICudaRowwiseImpl, public CRowwiseOperationDesc {
public:
	explicit CCudaRowwiseActivation( const CActivationDesc& desc ) : desc( desc ), dataSize( 0 ) {}

	TActivationFunction Type() const { return desc.GetType(); }

	// ICudaRowwiseImpl
	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int OutputSize() const override { return dataSize; }
	bool IsInPlace() const override { return true; }
	void Process( const CFloatHandle& input, const CFloatHandle& output ) const override;

private:
	CActivationDesc desc;
	int dataSize;
};

//---------------------------------------------------------------------------------------------------------------------

inline CBlobDesc CCudaRowwiseActivation::Reshape( const CBlobDesc& inputSize )
{
	dataSize = inputSize.BlobSize();
	return inputSize;
}

inline void CCudaRowwiseActivation::Process( const CFloatHandle& input, const CFloatHandle& output ) const
{
	IMathEngine& mathEngine = *input.GetMathEngine();
	switch( Type() ) {
		case AF_ELU:
			mathEngine.VectorELU( input, output, dataSize,
				desc.GetParam<CELUActivationParam>().Alpha );
			break;
		case AF_HardSigmoid:
			mathEngine.VectorHardSigmoid( input, output, dataSize,
				desc.GetParam<CHardSigmoidActivationParam>().Slope,
				desc.GetParam<CHardSigmoidActivationParam>().Bias );
			break;
		case AF_HardTanh:
			mathEngine.VectorHardTanh( input, output, dataSize );
			break;
		case AF_HSwish:
			mathEngine.VectorHSwish( input, output, dataSize );
			break;
		case AF_LeakyReLU:
			mathEngine.VectorLeakyReLU( input, output, dataSize,
				desc.GetParam<CLeakyReLUActivationParam>().Alpha );
			break;
		case AF_Linear:
		{
			CConstFloatHandle currInput = input;
			if( desc.GetParam<CLinearActivationParam>().Multiplier != 1.f ) {
				mathEngine.VectorMultiply( currInput, output, dataSize,
					desc.GetParam<CLinearActivationParam>().Multiplier );
				currInput = output;
			}
			if( desc.GetParam<CLinearActivationParam>().FreeTerm != 0.f ) {
				mathEngine.VectorAddValue( currInput, output, dataSize,
					desc.GetParam<CLinearActivationParam>().FreeTerm );
				currInput = output;
			}
			if( currInput != output ) {
				// Corner case: Linear( 1, 0 ), not in-place
				mathEngine.VectorCopy( output, currInput, dataSize );
			}
			break;
		}
		case AF_ReLU:
			mathEngine.VectorReLU( input, output, dataSize,
				desc.GetParam<CReLUActivationParam>().UpperThreshold );
			break;
		case AF_Sigmoid:
			mathEngine.VectorSigmoid( input, output, dataSize );
			break;
		case AF_Tanh:
			mathEngine.VectorTanh( input, output, dataSize );
			break;
		default:
			ASSERT_EXPR( false );
	}
}

} // namespace NeoML
