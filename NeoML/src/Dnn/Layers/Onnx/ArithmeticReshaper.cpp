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

#include <NeoML/Dnn/Layers/Onnx/ArithmeticReshaper.h>

namespace NeoML {

static const int ArithmeticReshaperVersion = 0;

void CArithmeticReshaper::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ArithmeticReshaperVersion );
	CBaseReshaper::Serialize( archive );
	archive.SerializeEnum( operation );
}

void CArithmeticReshaper::CalculateShapes()
{
	CheckInputs();
	CheckOutputs();
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "arithmetic operator with multiple outputs" );
	CheckArchitecture( inputShapeTensors[0] != nullptr, GetPath(), "input without shape tensor" );

	CFastArray<int, 8> resultSize;
	inputShapeTensors[0]->Size().CopyTo( resultSize );
	for( int inputIndex = 1; inputIndex < GetInputCount(); ++inputIndex ) {
		CheckArchitecture( inputShapeTensors[inputIndex] != nullptr, GetPath(), "input without shape tensor" );
		CShapeTensor::BroadcastSize( resultSize, inputShapeTensors[inputIndex]->Size() );
	}
	outputShapeTensors[0].Resize( resultSize );

	CFastArray<int, 8> coord;
	coord.Add( 0, outputShapeTensors[0].Rank() );
	for( int elemIndex = 0; elemIndex < outputShapeTensors[0].ElementCount(); ++elemIndex ) {
		outputShapeTensors[0].BroadcastingAt( coord ) = inputShapeTensors[0]->BroadcastingAt( coord );
		for( int inputIndex = 1; inputIndex < GetInputCount(); ++inputIndex ) {
			static_assert( static_cast<int>( TOperation::Count ) == 4, "TOperation::Count != 4" );
			switch( operation ) {
				case TOperation::Add:
					outputShapeTensors[0].BroadcastingAt( coord ) += inputShapeTensors[inputIndex]->BroadcastingAt( coord );
					break;
				case TOperation::Sub:
					outputShapeTensors[0].BroadcastingAt( coord ) -= inputShapeTensors[inputIndex]->BroadcastingAt( coord );
					break;
				case TOperation::Mul:
					outputShapeTensors[0].BroadcastingAt( coord ) *= inputShapeTensors[inputIndex]->BroadcastingAt( coord );
					break;
				case TOperation::Div:
					outputShapeTensors[0].BroadcastingAt( coord ) /= inputShapeTensors[inputIndex]->BroadcastingAt( coord );
					break;
				default:
					NeoAssert( false );
			}
		}
		if( !coord.IsEmpty() ) {
			coord.Last()++;
			int overflowIndex = coord.Size() - 1;
			while( overflowIndex > 0 && coord[overflowIndex] == resultSize[overflowIndex] ) {
				coord[overflowIndex - 1]++;
				coord[overflowIndex--] = 0;
			}
		}
	}
}

} // namespace NeoML
