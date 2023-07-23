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

CBlobDesc CCudaRowwiseActivation::Reshape( const CBlobDesc& inputSize )
{
	dataSize = inputSize.BlobSize();
	return inputSize;
}

inline void CCudaRowwiseActivation::Process( const CFloatHandle& input, const CFloatHandle& output ) const
{
	IMathEngine& mathEngine = *input.GetMathEngine();
	switch( desc.GetType() ) {
		case AF_ELU:
		{
			CFloatHandleStackVar alphaVar( mathEngine );
			alphaVar.SetValue( desc.GetParam<CELUActivationParam>().Alpha );
			mathEngine.VectorELU( input, output, dataSize, alphaVar.GetHandle() );
			break;
		}
		case AF_HardSigmoid:
		{
			CFloatHandleStackVar buff( mathEngine, 2 );
			buff.SetValueAt( 0, desc.GetParam<CHardSigmoidActivationParam>().Slope );
			CConstFloatHandle slope = buff.GetHandle();
			buff.SetValueAt( 1, desc.GetParam<CHardSigmoidActivationParam>().Bias );
			CConstFloatHandle bias = slope + 1;
			mathEngine.VectorHardSigmoid( input, output, dataSize, slope, bias );
			break;
		}
		case AF_HardTanh:
			mathEngine.VectorHardTanh( input, output, dataSize );
			break;
		case AF_HSwish:
			mathEngine.VectorHSwish( input, output, dataSize );
			break;
		case AF_LeakyReLU:
		{
			CFloatHandleStackVar alphaVar( mathEngine );
			alphaVar.SetValue( desc.GetParam<CLeakyReLUActivationParam>().Alpha );
			mathEngine.VectorLeakyReLU( input, output, dataSize, alphaVar.GetHandle() );
			break;
		}
		case AF_Linear:
		{
			CConstFloatHandle currInput = input;
			if( desc.GetParam<CLinearActivationParam>().Multiplier != 1.f ) {
				CFloatHandleStackVar mulVar( mathEngine );
				mulVar.SetValue( desc.GetParam<CLinearActivationParam>().Multiplier );
				mathEngine.VectorMultiply( currInput, output, dataSize, mulVar );
				currInput = output;
			}
			if( desc.GetParam<CLinearActivationParam>().FreeTerm != 0.f ) {
				CFloatHandleStackVar freeTermVar( mathEngine );
				freeTermVar.SetValue( desc.GetParam<CLinearActivationParam>().FreeTerm );
				mathEngine.VectorAddValue( currInput, output, dataSize, freeTermVar );
				currInput = output;
			}
			if( currInput != output ) {
				// Corner case: Linear( 1, 0 ), not in-place
				mathEngine.VectorCopy( output, currInput, dataSize );
			}
			break;
		}
		case AF_ReLU:
		{
			CFloatHandleStackVar thresholdVar( mathEngine );
			thresholdVar.SetValue( desc.GetParam<CReLUActivationParam>().UpperThreshold );
			mathEngine.VectorReLU( input, output, dataSize, thresholdVar );
			break;
		}
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
