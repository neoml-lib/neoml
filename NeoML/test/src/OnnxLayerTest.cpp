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

#include <common.h>
#pragma hdrstop

#include <TestFixture.h>

#include <NeoML/Dnn/Layers/Onnx/OnnxLayers.h>

using namespace NeoML;
using namespace NeoMLTest;

TEST( OnnxLayerTest, ReshapeCallTest )
{
	CRandom random( 0x654 );
	CDnn dnn( random, MathEngine() );

	// Net takes BD_BatchWidth of its input
	// Multiplies it by 2 and writes the result to sink
	CPtr<CSourceLayer> source = AddLayer<CSourceLayer>( "Source", dnn );
	CPtr<COnnxShapeLayer> shapeOp = AddLayer<COnnxShapeLayer>( "ShapeOp", { source } );
	shapeOp->TensorLayout() = { BD_BatchWidth };
	CPtr<COnnxSourceHelper> data = AddLayer<COnnxSourceHelper>( "Data", dnn );
	data->Blob() = CDnnBlob::CreateVector( MathEngine(), CT_Int, 1 );
	data->Blob()->Fill<int>( 2 );
	CPtr<COnnxEltwiseLayer> eltwise = AddLayer<COnnxEltwiseLayer>( "Eltwise", { shapeOp, data } );
	eltwise->SetOperation( COnnxEltwiseLayer::TOperation::Mul );
	CPtr<COnnxShapeToBlobLayer> toBlob = AddLayer<COnnxShapeToBlobLayer>( "ToBlob", { eltwise } );
	CPtr<CSinkLayer> sink = AddLayer<CSinkLayer>( "Sink", { toBlob } );

	// Step 1: check initial run
	source->SetBlob( CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 3, 5 ) );
	dnn.RunOnce();
	ASSERT_EQ( 6, sink->GetBlob()->GetData<int>().GetValue() );

	// Step 2: change the shape of the input
	source->SetBlob( CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 5, 3 ) );
	dnn.RunOnce();
	ASSERT_EQ( 10, sink->GetBlob()->GetData<int>().GetValue() );

	// Step 3: change the multiplier coeff
	data->Blob()->Fill<int>( 3 );
	dnn.RunOnce();
	ASSERT_EQ( 15, sink->GetBlob()->GetData<int>().GetValue() );

	// Step 4: change the operation from Mul to Add
	eltwise->SetOperation( COnnxEltwiseLayer::TOperation::Add );
	dnn.RunOnce();
	ASSERT_EQ( 8, sink->GetBlob()->GetData<int>().GetValue() );

	// Step 5: change addendum
	data->Blob()->Fill<int>( -10 );
	dnn.RunOnce();
	ASSERT_EQ( -5, sink->GetBlob()->GetData<int>().GetValue() );

	// Step 6: change the shape of the input
	source->SetBlob( CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, 12, 1 ) );
	dnn.RunOnce();
	ASSERT_EQ( 2, sink->GetBlob()->GetData<int>().GetValue() );
}

