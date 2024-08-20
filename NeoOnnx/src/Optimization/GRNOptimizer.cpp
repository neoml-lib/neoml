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

#include <common.h>
#pragma hdrstop

#include <initializer_list>

#include <NeoOnnx/NeoOnnxImport.h>

#include "Graph.h"
#include "GRNOptimizer.h"
#include <NeoML/Dnn/Layers/GlobalMeanPoolingLayer.h>
#include <NeoML/Dnn/Layers/GlobalSumPoolingLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxTransformHelper.h>
#include <TensorUtils.h>

using namespace NeoML;

namespace NeoOnnx {

class CGrnLayoutValidator : public ITensorLayoutValidator {
public:
	CGrnLayoutValidator( const CFastArray<int, 2>& _geomAxes, int channelAxis ) :
		channelAxis( channelAxis ) { _geomAxes.CopyTo( geomAxes ); }

	bool operator()( const CTensorLayout& layout ) const override;

private:
	CFastArray<int, 2> geomAxes;
	int channelAxis;
};

bool CGrnLayoutValidator::operator()( const CTensorLayout& layout ) const
{
	for( int i = 0; i < layout.Size(); ++i ) {
		if( i == channelAxis && layout[i] != BD_Channels ) {
			return false; // layout[channelAxis] must be BD_Channels
		} else if( geomAxes.Find( i ) != NotFound && ( layout[i] < BD_Height || layout[i] == BD_Channels ) ) {
			return false; // layout[geomAxes] must be BD_Height, BD_Width or BD_Depth
		} else if( i != channelAxis && geomAxes.Find( i ) == NotFound && layout[i] > BD_ListSize ) {
			return false; // layout[otherAxes] must be BD_BatchLength, BD_BatchWidth or BD_ListSize
		}
	}
	return true;
}

namespace optimization {

// Checks that the following eltwise operation is
//    1. of a given type
//    2. has 2 inputs
//    3. one of its inputs is a data layer with float blob of given size (0 means no size restrictions along dim)
//    4. other input is of a TNonDataLayer type
// Returns true if all conditions are met
// Otherwise returns false
template<typename TNonDataLayer>
static bool processGrnEltwiseOp( COnnxEltwiseLayer& eltwise, COnnxEltwiseLayer::TOperation operation,
	std::initializer_list<int> expectedSize, CGraph& graph, CLayerOutput<CDataLayer>& data,
	CLayerOutput<TNonDataLayer>& otherInput )
{
	NeoAssert( expectedSize.size() == static_cast<size_t>( BD_Count ) );

	if( eltwise.GetOperation() != operation || graph.GetInputCount( eltwise ) != 2 ) {
		return false;
	}

	if( !graph.SelectBothConnectedOutputs( eltwise, data, otherInput, true ) ) {
		return false;
	}

	const CPtr<CDnnBlob>& blob = data.Layer->GetBlob();
	if( blob->GetDataType() != CT_Float ) {
		return false;
	}

	const int* expectedSizePtr = expectedSize.begin();
	for( int i = 0; i < BD_Count; ++i ) {
		if( expectedSizePtr[i] > 0 && blob->DimSize( i ) != expectedSizePtr[i] ) {
			return false;
		}
	}

	return true;
}

// Checks that 2 transformers are the inversions of each other
static bool areGrnTransformersInverted( const COnnxTransformHelper& first, const COnnxTransformHelper& second )
{
	auto isInv = [] ( const COnnxTransformHelper& orig, const COnnxTransformHelper& inv ) -> bool
	{
		for( TBlobDim dim = BD_Channels; dim != BD_Count; ++dim ) {
			TBlobDim rule = orig.GetRule( dim );
			if( rule != BD_Count && inv.GetRule( rule ) != dim ) {
				return false;
			}
		}

		return true;
	};

	return isInv( first, second ) && isInv( second, first );
}

// Gets axes indices which were reduced by global poolings for given layout
static void getGrnPooledAxes( const CFastArray<TBlobDim, 8>& layout, CFastArray<int, 2>& axes )
{
	axes.Empty();
	for( int i = 0; i < layout.Size(); ++i ) {
		if( layout[i] >= BD_Height && layout[i] != BD_Channels ) {
			axes.Add( i );
		}
	}
}

// Replaces grn starting in current layer and where residual connection is at index'th input
static bool replaceGrn( CGraph& graph, CBaseLayer& layer, int residualInputIndex )
{
	NeoPresume( graph.HasLayer( &layer ) );
	NeoPresume( residualInputIndex == 0 || residualInputIndex == 1 );
	graph.ClearSelection();

	COnnxEltwiseLayer* residualAdd = dynamic_cast<COnnxEltwiseLayer*>( &layer );
	if( residualAdd == nullptr || residualAdd->GetOperation() != COnnxEltwiseLayer::TOperation::Add
		|| residualAdd->GetInputCount() != 2 )
	{
		return false;
	}

	graph.SelectLayer( *residualAdd );
	CLayerOutput<> blockData = graph.GetConnectedOutput( *residualAdd, residualInputIndex );
	COnnxEltwiseLayer* biasAdd = graph.SelectConnectedOutput<COnnxEltwiseLayer>( *residualAdd, 1 - residualInputIndex,
		true ).Layer;

	CLayerOutput<CDataLayer> biasData;
	CLayerOutput<COnnxEltwiseLayer> scaleMul;
	if( biasAdd == nullptr || !processGrnEltwiseOp( *biasAdd, COnnxEltwiseLayer::TOperation::Add,
		{ 1, 1, 1, 1, 1, 1, 0 }, graph, biasData, scaleMul ) )
	{
		return false;
	}
	
	CLayerOutput<CDataLayer> scaleData;
	CLayerOutput<COnnxEltwiseLayer> mulXbyNx; // see formula in CGrnLayer comment
	if( !processGrnEltwiseOp( *scaleMul.Layer, COnnxEltwiseLayer::TOperation::Mul,
		{ 1, 1, 1, 1, 1, 1, 0 }, graph, scaleData, mulXbyNx ) )
	{
		return false;
	}

	if( graph.GetInputCount( *mulXbyNx.Layer ) != 2 ) {
		return false;
	}

	int blockDataInputIndex = NotFound;
	for( int i = 0; i < 2; ++i ) {
		if( graph.GetConnectedOutput( *mulXbyNx.Layer, i ) == blockData ) {
			blockDataInputIndex = i;
			break;
		}
	}

	if( blockDataInputIndex == NotFound ) {
		return false;
	}

	COnnxTransformHelper* nxTransform = graph.SelectConnectedOutput<COnnxTransformHelper>( *mulXbyNx.Layer,
		1 - blockDataInputIndex, true ).Layer;

	COnnxEltwiseLayer* gxDivByMean = nullptr;
	if( nxTransform == nullptr ) {
		gxDivByMean = graph.SelectConnectedOutput<COnnxEltwiseLayer>( *mulXbyNx.Layer,
			1 - blockDataInputIndex, true ).Layer;
	} else {
		gxDivByMean = graph.SelectTheOnlyConnectedOutput<COnnxEltwiseLayer>( *nxTransform, true );
	}

	CLayerOutput<COnnxTransformHelper> meanTransform;
	CLayerOutput<CPowerLayer> sqrt;
	if( gxDivByMean == nullptr || gxDivByMean->GetOperation() != COnnxEltwiseLayer::TOperation::Div
		|| !graph.SelectBothConnectedOutputs( *gxDivByMean, meanTransform, sqrt, false ) )
	{
		return false;
	}

	COnnxEltwiseLayer* addEps = graph.SelectTheOnlyConnectedOutput<COnnxEltwiseLayer>( *meanTransform.Layer, true );
	CLayerOutput<CDataLayer> epsData;
	CLayerOutput<CGlobalMeanPoolingLayer> gxMean;
	if( addEps == nullptr || !processGrnEltwiseOp( *addEps, COnnxEltwiseLayer::TOperation::Add,
		{ 1, 1, 1, 1, 1, 1, 1 }, graph, epsData, gxMean ) )
	{
		return false;
	}

	COnnxTransformHelper* preMeanTransform = graph.SelectTheOnlyConnectedOutput<COnnxTransformHelper>( *gxMean.Layer,
		true );
	if( preMeanTransform == nullptr || !areGrnTransformersInverted( *preMeanTransform, *meanTransform.Layer )
		|| graph.GetInputCount( *preMeanTransform ) != 1 || graph.GetConnectedOutput( *preMeanTransform, 0 ) != sqrt
		|| sqrt.Layer->GetExponent() != 0.5f )
	{
		return false;
	}

	CGlobalSumPoolingLayer* sqrSum = graph.SelectTheOnlyConnectedOutput<CGlobalSumPoolingLayer>( *sqrt.Layer, true );
	if( sqrSum == nullptr ) {
		return false;
	}

	CPowerLayer* sqr = nullptr;
	if( nxTransform != nullptr ) {
		COnnxTransformHelper* sqrTransform = graph.SelectTheOnlyConnectedOutput<COnnxTransformHelper>( *sqrSum, true );
		if( sqrTransform == nullptr || !areGrnTransformersInverted( *sqrTransform, *nxTransform ) ) {
			return false;
		}
		sqr = graph.SelectTheOnlyConnectedOutput<CPowerLayer>( *sqrTransform, true );
	} else {
		sqr = graph.SelectTheOnlyConnectedOutput<CPowerLayer>( *sqrSum, true );
	}

	if( sqr == nullptr || sqr->GetExponent() != 2.f ) {
		return false;
	}

	// For the reason unknown there is an Abs operation before the square
	CAbsLayer* abs = graph.SelectTheOnlyConnectedOutput<CAbsLayer>( *sqr, true );
	if( abs == nullptr || graph.GetInputCount( *abs ) != 1
		|| graph.GetConnectedOutput( *abs, 0 ) != blockData )
	{
		return false;
	}

	// Figure out which ONNX axes were reduced during first pooling
	CFastArray<int, 2> geomAxes;
	getGrnPooledAxes( preMeanTransform->InputLayout(), geomAxes );
	CFastArray<int, 2> channelAxes;
	getGrnPooledAxes( preMeanTransform->OutputLayout(), channelAxes );
	if( channelAxes.Size() != 1 || geomAxes.Find( channelAxes[0] ) != NotFound ) {
		return false;
	}

	CTensorLayout inputLayout;
	if( nxTransform == nullptr ) {
		preMeanTransform->InputLayout().CopyTo( inputLayout );
	} else {
		nxTransform->OutputLayout().CopyTo( inputLayout );
	}

	// Converting grn input into layout compatible with grn
	CTensorLayout grnLayout;
	CLayerOutput<> currOutput = ConvertTensor( blockData, inputLayout, CGrnLayoutValidator( geomAxes, channelAxes[0] ),
		graph, grnLayout );

	CPtr<CGrnLayer> grn = new CGrnLayer( graph.MathEngine() );
	grn->SetName( graph.GetUniqueName( "GRN_" ) );
	grn->SetEpsilon( epsData.Layer->GetBlob()->GetData<const float>().GetValue() );
	grn->SetScale( scaleData.Layer->GetBlob() );
	grn->SetBias( biasData.Layer->GetBlob() );
	graph.AddLayer( *grn );
	graph.Connect( *grn, 0, *currOutput.Layer, currOutput.Index );

	CTensorLayout outputLayout;
	currOutput = ConvertTensor( CLayerOutput<>( grn, 0 ), grnLayout, CTensorLayoutMatchValidator( inputLayout ),
		graph, outputLayout );
	NeoAssert( outputLayout == inputLayout );

	graph.SwitchOutputs( *residualAdd, 0, *currOutput.Layer, currOutput.Index );
	graph.DeleteSelectedLayers();

	return true;
}

int OptimizeGRN( CGraph& graph )
{
	int result = 0;

	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );
	for( CBaseLayer* layer : layers ) {
		if( !graph.HasLayer( layer ) ) {
			continue;
		}

		if( replaceGrn( graph, *layer, 0 ) || replaceGrn( graph, *layer, 1 ) ) {
			++result;
		}
	}

	graph.ClearSelection();
	return result;
}

} // namespace optimization

} // namespace NeoOnnx
