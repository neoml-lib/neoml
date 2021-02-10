/* Copyright © 2017-2020 ABBYY Production LLC

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

#include "LrnNode.h"
#include "GraphCache.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CLrnNode::CLrnNode( int nodeIndex, const onnx::NodeProto& lrn, int opsetVersion ) :
	COpNode( nodeIndex, lrn, opsetVersion ),
	alpha( Attributes.GetOptionalFloat( "alpha", 1e-4f ) ),
	beta( Attributes.GetOptionalFloat( "beta", 0.75f ) ),
	bias( Attributes.GetOptionalFloat( "bias", 1.f ) ),
	size( Attributes.GetRequiredInt( "size" ) )
{
	// Introduce in opset v1
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", lrn );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", lrn );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", lrn );
}

void CLrnNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& /* mathEngine */ )
{
	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
	tensors[Output[0]] = tensors[Input[0]];
}

void CLrnNode::LabelTensorDims( const CTensorCache &tensors, CDimCache &dims )
{
	CTensorDim tensorDims;
	if( tensors[Input[0]].Shape.Size() == 4 ) {
		tensorDims.Add( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width } );
	} else {
		tensorDims.Add( { BD_BatchWidth, BD_Channels, BD_Depth, BD_Height, BD_Width } );
	}

	CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, tensorDims, dims[Output[0]] ),
		"labeling output dimensions failed", OnnxNode );
	CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, tensorDims, dims[Input[0]] ),
		"labeling output dimensions failed", OnnxNode );
}

void CLrnNode::AddLayers( const CGraph& /* graph */, const CTensorCache& /* tensors */, const CDimCache& /* dims */,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<CLrnLayer> lrnLayer = new CLrnLayer( dnn.GetMathEngine() );
	lrnLayer->SetName( Name );
	lrnLayer->SetWindowSize( size );
	lrnLayer->SetBias( bias );
	lrnLayer->SetAlpha( alpha );
	lrnLayer->SetBeta( beta );
	lrnLayer->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *lrnLayer );
	neoMLLinks[Output[0]] = CNeoMLLink( lrnLayer.Ptr(), 0 );
}

} // namespace NeoOnnx