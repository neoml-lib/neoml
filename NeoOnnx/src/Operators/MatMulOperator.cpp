/* Copyright © 2017-2024 ABBYY

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

#include "../common.h"
#pragma hdrstop

#include <algorithm>

#include "MatMulOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

using namespace NeoML;

namespace NeoOnnx {

CMatMulOperator::CMatMulOperator( const onnx::NodeProto& matMul, int opsetVersion ) :
	CLayerOperator( matMul, opsetVersion )
{
	// v1 - original
	// v9 - integer data is supported
	// v13 - bfloat16 data is supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 2, "operator must have 2 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CMatMulOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );
	CheckNoShapeInputs( inputs );

	// NeoML doesn't support batch broadcast of the first argument in matrix multiplication
	CheckNeoOnnxSupport( inputs[0]->DimCount() <= 5 && inputs[1]->DimCount() <= 5, "Too many batch dimensions", *this );

	CPtr<const CUserTensor> first = prepareTensor( *inputs[0], true, dnn );
	CPtr<const CUserTensor> second = prepareTensor( *inputs[1], false, dnn );

	CPtr<CMatrixMultiplicationLayer> matmul = new CMatrixMultiplicationLayer( dnn.GetMathEngine() );
	matmul->SetName( Name() );
	matmul->Connect( 0, *first->Layer(), first->OutputIndex() );
	matmul->Connect( 1, *second->Layer(), second->OutputIndex() );
	dnn.AddLayer( *matmul );

	CTensorLayout outputLayout;
	if( first->DimCount() == 1 && second->DimCount() <= 2 ) {
		outputLayout = { BD_Channels };
	} if( second->DimCount() == 1 && first->DimCount() <= 2 ) {
		outputLayout = { BD_Height };
	} else {
		outputLayout = { BD_BatchLength, BD_BatchWidth, BD_ListSize, BD_Height, BD_Channels };
		const int outputDimCount = std::max<int>( first->DimCount(), second->DimCount() );
		outputLayout.DeleteAt( 0, outputLayout.Size() - outputDimCount );
	}

	outputs.Add( new CUserTensor( outputLayout, CLayerOutput( matmul, 0 ) ) );
}

// Prepares tensor for CMatrixMultiplicationLayer
// It puts matrix height into BD_Height
// It puts matrix width into BD_Channels
CPtr<const CUserTensor> CMatMulOperator::prepareTensor( const CTensorBase& tensor, bool isFirstArg, CDnn& dnn ) const
{
	CPtr<const CTensorBase> currTensor = &tensor;
	CTensorLayout outputLayout( { BD_BatchLength, BD_BatchWidth, BD_ListSize, BD_Height, BD_Channels } );
	if( tensor.DimCount() == 1 && !isFirstArg ) {
		outputLayout = { BD_Height };
	} else {
		outputLayout.DeleteAt( 0, outputLayout.Size() - currTensor->DimCount() );
	}

	return AsUserTensor( *ConvertTensor( *currTensor, outputLayout ),
		Name() + ( isFirstArg ? "_Input0" : "_Input1" ), dnn );
}

} // namespace NeoOnnx
