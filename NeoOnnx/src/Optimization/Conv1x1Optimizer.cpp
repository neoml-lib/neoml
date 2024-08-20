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

#include "common.h"
#pragma hdrstop

#include "Graph.h"
#include "GELUOptimizer.h"
#include <NeoML/Dnn/Layers/DataLayer.h>
#include <NeoML/Dnn/Layers/MatrixMultiplicationLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>

using namespace NeoML;

namespace NeoOnnx {

namespace optimization {

static CPtr<CDnnBlob> getConv1x1Filter( CDataLayer* dataLayer )
{
	if( dataLayer == nullptr ) {
		return nullptr;
	}

	const CPtr<CDnnBlob>& dataBlob = dataLayer->GetBlob();
	if( dataBlob == nullptr || dataBlob->GetDataType() != CT_Float || dataBlob->GetObjectCount() != 1 ) {
		return nullptr;
	}

	// MatMul layer expect matrix height in blob->GetGeometricalSize()
	// Conv1x1 filter is stored in transposed order and it's height must be along blob->GetObjectSize()
	IMathEngine& mathEngine = dataBlob->GetMathEngine();
	CPtr<CDnnBlob> resultBlob = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1,
		dataBlob->GetChannelsCount(), dataBlob->GetGeometricalSize() );
	mathEngine.TransposeMatrix( 1, dataBlob->GetData(), dataBlob->GetGeometricalSize(), 1, dataBlob->GetChannelsCount(),
		1, resultBlob->GetData(), resultBlob->GetDataSize() );
	return resultBlob;
}

static CPtr<CDnnBlob> getConv1x1FreeTerm( CDataLayer* dataLayer, int expectedSize )
{
	if( dataLayer == nullptr ) {
		return nullptr;
	}

	const CPtr<CDnnBlob>& dataBlob = dataLayer->GetBlob();
	if( dataBlob == nullptr || dataBlob->GetDataType() != CT_Float
		|| dataBlob->GetDataSize() != dataBlob->GetChannelsCount()
		|| dataBlob->GetDataSize() != expectedSize )
	{
		return nullptr;
	}

	return dataBlob->GetCopy();
}

// Replaces current matmul layer with Conv1x1 if possible
// Returns true if succeeded
// Returns false otherwise
static bool replaceMatMulWithConv( NeoML::optimization::CGraph& graph, CBaseLayer* layer )
{
	CMatrixMultiplicationLayer* matmul = dynamic_cast<CMatrixMultiplicationLayer*>( layer );
	if( matmul == nullptr || graph.GetInputCount( *matmul ) != 2 ) {
		return false;
	}
	graph.ClearSelection();
	graph.SelectLayer( *matmul );

	CLayerOutput<CDataLayer> filterData;
	int filterDataIndex = NotFound;
	for( int i = 0; i < 2; ++i ) {
		filterData = graph.SelectConnectedOutput<CDataLayer>( *matmul, i, true );
		filterDataIndex = i;
		if( filterData.Layer != nullptr ) {
			break;
		}
	}

	CPtr<CDnnBlob> filter = getConv1x1Filter( filterData.Layer );
	if( filter == nullptr ) {
		return false;
	}

	CPtr<CConvLayer> conv = new CConvLayer( graph.MathEngine() );
	conv->SetName( graph.GetUniqueName( "Conv_" ) );
	conv->SetFilterCount( filter->GetObjectCount() );
	conv->SetZeroFreeTerm( true );
	conv->SetFilterHeight( 1 );
	conv->SetFilterWidth( 1 );
	conv->SetStrideHeight( 1 );
	conv->SetStrideWidth( 1 );
	conv->SetPaddingHeight( 0 );
	conv->SetPaddingWidth( 0 );
	conv->SetFilterData( filter );
	graph.AddLayer( *conv );

	CLayerOutput<> convData = graph.GetConnectedOutput( *matmul, 1 - filterDataIndex );
	graph.Connect( *conv, 0, *convData.Layer, convData.Index );
	graph.SwitchOutputs( *matmul, 0, *conv, 0 );
	graph.DeleteSelectedLayers();

	return true;
}

// Merges current add operation into convolution (if possible)
void mergeAddOpIntoConv( NeoML::optimization::CGraph& graph, CBaseLayer* layer )
{
	COnnxEltwiseLayer* biasAdd = dynamic_cast<COnnxEltwiseLayer*>( layer );
	if( biasAdd == nullptr || biasAdd->GetOperation() != COnnxEltwiseLayer::TOperation::Add ) {
		return;
	}

	graph.ClearSelection();
	graph.SelectLayer( *biasAdd );
	CLayerOutput<CDataLayer> biasData;
	CLayerOutput<CConvLayer> conv;
	if( !graph.SelectBothConnectedOutputs( *biasAdd, biasData, conv, true ) ) {
		return;
	}

	graph.UndoSelectLayer( *conv.Layer );
	CPtr<CDnnBlob> freeTerm = getConv1x1FreeTerm( biasData.Layer, conv.Layer->GetFilterCount() );
	if( freeTerm == nullptr ) {
		return;
	}

	conv.Layer->SetZeroFreeTerm( false );
	CPtr<CDnnBlob> prevFreeTerm = conv.Layer->GetFreeTermData();
	if( prevFreeTerm != nullptr ) {
		NeoAssert( prevFreeTerm->GetDataSize() == freeTerm->GetDataSize() );
		graph.MathEngine().VectorAdd( freeTerm->GetData(), prevFreeTerm->GetData(), freeTerm->GetData(),
			freeTerm->GetDataSize() );
	}
	conv.Layer->SetFreeTermData( freeTerm );

	graph.SwitchOutputs( *biasAdd, 0, *conv.Layer, conv.Index );
	graph.DeleteSelectedLayers();
}

int OptimizeConv1x1( NeoML::optimization::CGraph& graph )
{
	int result = 0;
	// Step 1: replace MatMul with 1x1 Conv
	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );
	for( CBaseLayer* layer : layers ) {
		NeoAssert( graph.HasLayer( layer ) );
		if( replaceMatMulWithConv( graph, layer ) ) {
			++result;
		} else {
			mergeAddOpIntoConv( graph, layer );
		}
	}

	graph.ClearSelection();
	return result;
}

} // namespace optimization

} // namespace NeoOnnx
