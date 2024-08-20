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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/Onnx/OnnxShapeLayer.h>

#include <limits.h>
#include <algorithm>

namespace NeoML {

COnnxShapeLayer::COnnxShapeLayer( IMathEngine& mathEngine ) :
	COnnxLayerBase( mathEngine, "OnnxShapeLayer" ),
	startAttr( 0 ),
	endAttr( INT_MAX )
{
}

static const int OnnxShapeLayerVersion = 1;

void COnnxShapeLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( OnnxShapeLayerVersion );
	COnnxLayerBase::Serialize( archive );
	tensorLayout.Serialize( archive );
	
	if( version == 0 && archive.IsLoading() ) {
		startAttr = 0;
		endAttr = INT_MAX;
	}

	if( version > 0 ) {
		archive.Serialize( startAttr );
		archive.Serialize( endAttr );
	}
}

void COnnxShapeLayer::CalculateShapes()
{
	CheckInput1();
	CheckLayerArchitecture( GetOutputCount() == 1, "layer must have 1 output" );

	const int start = std::max<int>( 0, startAttr < 0 ? startAttr + tensorLayout.Size() : startAttr );
	const int end = std::min<int>( endAttr < 0 ? endAttr + tensorLayout.Size() : endAttr, tensorLayout.Size() );
	outputShapeBlobs[0] = CDnnBlob::CreateVector( MathEngine(), CT_Int, end - start );
	CDnnBlobBuffer<int> outputBuff( *outputShapeBlobs[0], TDnnBlobBufferAccess::Write );
	for( int dimIndex = start; dimIndex < end; ++dimIndex ) {
		outputBuff[dimIndex - start] = inputDescs[0].DimSize( tensorLayout[dimIndex] );
	}
}

} // namespace NeoML
