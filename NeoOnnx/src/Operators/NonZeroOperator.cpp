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

#include "../common.h"
#pragma hdrstop

#include "onnx.pb.h"

#include "NonZeroOperator.h"
#include "NeoOnnxCheck.h"

#include <NeoML/Dnn/Layers/Onnx/OnnxNonZeroLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxConstantOfShapeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxRangeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxSourceHelper.h>

namespace NeoOnnx {

static CPtr<const CShapeTensor> addNonZeroOpSourceHelper( const CString& name, CDnn& dnn, int value )
{
	CPtr<COnnxSourceHelper> source = new COnnxSourceHelper( dnn.GetMathEngine() );
	source->SetName( name );
	source->Blob() = CDnnBlob::CreateDataBlob( dnn.GetMathEngine(), CT_Int, 1, 1, 1 );
	source->Blob()->Fill<int>( value );
	dnn.AddLayer( *source );
	return new CShapeTensor( CTensorLayout( { BD_BatchLength } ), CTensorShape( { 1 } ), CLayerOutput( source, 0 ) );
}

//---------------------------------------------------------------------------------------------------------------------

CNonZeroOperator::CNonZeroOperator( const onnx::NodeProto& nonZero, int opsetVersion ) :
	CLayerOperator( nonZero, opsetVersion )
{
	// v9 - original
	// v13 - new data types are supported
	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CNonZeroOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );

	if( inputs[0]->Type() == TTensorType::User ) {
		const CUserTensor& input = dynamic_cast<const CUserTensor&>( *inputs[0] );
		COnnxConstantOfShapeLayer* constantOfShapeLayer = dynamic_cast<COnnxConstantOfShapeLayer*>( input.Layer() );
		CheckNeoOnnxSupport( constantOfShapeLayer != nullptr,
			"NonZero operator supports user input only from ConstantOfShape", *this );
		if( constantOfShapeLayer->GetValue().GetDataType() == CT_Float ) {
			CheckNeoOnnxSupport( constantOfShapeLayer->GetValue().GetData().GetValue() != 0,
				"NonZero operator supprots user input only with ConstantOfShape( value != 0 )", *this );
		} else {
			CheckNeoOnnxSupport( constantOfShapeLayer->GetValue().GetData<int>().GetValue() != 0,
				"NonZero operator supprots user input only with ConstantOfShape( value != 0 )", *this );
		}
		// Sink in order to avoid hanging output
		Sink( constantOfShapeLayer, Name() + "_SafeSink" );

		// Some exporters use 1-dimensional ConstantOfShape + NonZero which is equivalent of Range
		CPtr<COnnxRangeLayer> rangeLayer = new COnnxRangeLayer( dnn.GetMathEngine() );
		rangeLayer->SetName( Name() );
		
		CPtr<const CShapeTensor> startTensor = addNonZeroOpSourceHelper( Name() + "_Start", dnn, 0 );
		rangeLayer->Connect( 0, *startTensor->Layer(), startTensor->OutputIndex() );

		rangeLayer->Connect( 1, constantOfShapeLayer->GetInputName( 0 ),
			constantOfShapeLayer->GetInputOutputNumber( 0 ) );

		CPtr<const CShapeTensor> deltaTensor = addNonZeroOpSourceHelper( Name() + "_Delta", dnn, 1 );
		rangeLayer->Connect( 2, *deltaTensor->Layer(), deltaTensor->OutputIndex() );

		dnn.AddLayer( *rangeLayer );

		// COnnxRangeLayer will return the indices of non-zero elements in BD_BatchLength (other dims will be 1)
		// But NonZero specification requires 2-dimensional output of Nx1 
		outputs.Add( new CUserTensor( CTensorLayout( { BD_BatchLength, BD_BatchWidth } ),
			CLayerOutput( rangeLayer, 0 ) ) );
		return;
	}

	CPtr<COnnxNonZeroLayer> nonZeroLayer = new COnnxNonZeroLayer( dnn.GetMathEngine() );
	nonZeroLayer->SetName( Name() );
	inputs[0]->Layout().CopyTo( nonZeroLayer->InputLayout() );
	dnn.AddLayer( *nonZeroLayer );

	CPtr<const CShapeTensor> input = AsShapeTensor( *inputs[0], Name() + "_Source", dnn );
	nonZeroLayer->Connect( 0, *input->Layer(), input->OutputIndex() );
	outputs.Add( new CUserTensor( input->Layout(), CLayerOutput( nonZeroLayer, 0 ) ) );
}

} // namespace NeoOnnx
