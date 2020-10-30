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

#include "ActivationNode.h"
#include "GraphCache.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CActivationNodeBase::CActivationNodeBase( int nodeIndex, const onnx::NodeProto& onnxNode,
		int opsetVersion, TActivationFunction _activation ) :
	COpNode( nodeIndex, onnxNode, opsetVersion ),
	activation( _activation )
{
}

void CActivationNodeBase::CalcOutputTensors( CTensorCache& tensors, IMathEngine& /* mathEngine */ )
{
	CheckOnnxNode();
	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
	tensors[Input[0]].Shape.CopyTo( tensors[Output[0]].Shape );
}

void CActivationNodeBase::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	if( !dims[Input[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, dims[Input[0]], dims[Output[0]] ),
			"labeling output dimensions failed", OnnxNode );
	}

	if( !dims[Output[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, dims[Output[0]], dims[Input[0]] ),
			"labeling input dimensions failed", OnnxNode );
	}
}

void CActivationNodeBase::AddLayers( const CGraph& /* graph */, const CTensorCache& tensors, const CDimCache& /* dims */,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<CBaseLayer> activationLayer = CreateActivationLayer( dnn.GetMathEngine(), activation );
	activationLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );
	SetLayerParams( tensors, activationLayer );
	activationLayer->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *activationLayer );
	neoMLLinks[Output[0]] = CNeoMLLink( activationLayer, 0 );
}

//---------------------------------------------------------------------------------------------------------------------

CAbsNode::CAbsNode( int nodeIndex, const onnx::NodeProto& abs, int opsetVersion ) :
	CActivationNodeBase( nodeIndex, abs, opsetVersion, AF_Abs )
{
}

void CAbsNode::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes and added new data types support
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", OnnxNode );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", OnnxNode );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", OnnxNode );
}

//---------------------------------------------------------------------------------------------------------------------

CClipNode::CClipNode( int nodeIndex, const onnx::NodeProto& clip, int opsetVersion ) :
	CActivationNodeBase( nodeIndex, clip, opsetVersion, AF_ReLU )
{
}

void CClipNode::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes
	// v11 - min/max values were moved from attributes to additional inputs
	// v12 - new data types supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", OnnxNode );

	if( OpsetVersion < 11 ) {
		CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", OnnxNode );
	} else {
		CheckOnnxProtocol( InputCount() >= 1 || InputCount() <= 3, "node must have from 1 up to 3 inputs", OnnxNode );
	}

	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", OnnxNode );
}

void CClipNode::SetLayerParams( const CTensorCache& tensors, CBaseLayer* layer ) const
{
	CReLULayer* relu = dynamic_cast<CReLULayer*>( layer );
	CheckNeoOnnxInternal( relu != nullptr, "wrong layer class", OnnxNode );

	float minValue = -FLT_MAX;
	float maxValue = FLT_MAX;
	if( OpsetVersion < 11 ) {
		minValue = Attributes.GetOptionalFloat( "min", -FLT_MAX );
		maxValue = Attributes.GetOptionalFloat( "max", FLT_MAX );
	} else if( InputCount() > 1 ) {
		const CDnnBlob* minValueBlob = tensors[Input[1]].Data;
		CheckNeoOnnxSupport( minValueBlob != nullptr, "user-provided clip min value", OnnxNode );
		if( minValueBlob->GetDataType() == CT_Float ) {
			minValue = minValueBlob->GetData<float>().GetValue();
		} else {
			minValue = static_cast<float>( minValueBlob->GetData<int>().GetValue() );
		}

		if( InputCount() > 2 ) {
			const CDnnBlob* maxValueBlob = tensors[Input[2]].Data;
			CheckNeoOnnxSupport( maxValueBlob != nullptr, "user-provided clip min value", OnnxNode );
			if( maxValueBlob->GetDataType() == CT_Float ) {
				maxValue = maxValueBlob->GetData<float>().GetValue();
			} else {
				maxValue = static_cast<float>( maxValueBlob->GetData<int>().GetValue() );
			}
		}
	}

	CheckNeoOnnxSupport( minValue == 0, "Clip with non-zero min value", OnnxNode );
	if( maxValue != FLT_MAX ) {
		relu->SetUpperThreshold( maxValue );
	}
}

//---------------------------------------------------------------------------------------------------------------------

CEluNode::CEluNode( int nodeIndex, const onnx::NodeProto& elu, int opsetVersion ) :
	CActivationNodeBase( nodeIndex, elu, opsetVersion, AF_ELU )
{
}

void CEluNode::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", OnnxNode );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", OnnxNode );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", OnnxNode );
}

//---------------------------------------------------------------------------------------------------------------------

CLeakyReluNode::CLeakyReluNode( int nodeIndex, const onnx::NodeProto& leakyRelu, int opsetVersion ) :
	CActivationNodeBase( nodeIndex, leakyRelu, opsetVersion, AF_LeakyReLU )
{
}

void CLeakyReluNode::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", OnnxNode );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", OnnxNode );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", OnnxNode );
}

void CLeakyReluNode::SetLayerParams( const CTensorCache& /* tensors */, CBaseLayer* layer ) const
{
	CLeakyReLULayer* leakyReLU = dynamic_cast<CLeakyReLULayer*>( layer );
	CheckNeoOnnxInternal( leakyReLU != nullptr, "wrong layer class", OnnxNode );
	leakyReLU->SetAlpha( Attributes.GetOptionalFloat( "alpha", 0.f ) );
}

//---------------------------------------------------------------------------------------------------------------------

CHardSigmoidNode::CHardSigmoidNode( int nodeIndex, const onnx::NodeProto& hardSigmoid, int opsetVersion ) :
	CActivationNodeBase( nodeIndex, hardSigmoid, opsetVersion, AF_HardSigmoid )
{
}

void CHardSigmoidNode::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", OnnxNode );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", OnnxNode );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", OnnxNode );
}

void CHardSigmoidNode::SetLayerParams( const CTensorCache& /* tensors */, CBaseLayer* layer ) const
{
	CHardSigmoidLayer* hardSigmoid = dynamic_cast<CHardSigmoidLayer*>( layer );
	CheckNeoOnnxInternal( hardSigmoid != nullptr, "wrong layer class", OnnxNode );

	const float alpha = Attributes.GetOptionalFloat( "alpha", 0.2f );
	const float beta = Attributes.GetOptionalFloat( "beta", 0.5f );

	hardSigmoid->SetSlope( alpha );
	hardSigmoid->SetBias( beta );
}

//---------------------------------------------------------------------------------------------------------------------

CReluNode::CReluNode( int nodeIndex, const onnx::NodeProto& relu, int opsetVersion ) :
	CActivationNodeBase( nodeIndex, relu, opsetVersion, AF_ReLU )
{
}

void CReluNode::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", OnnxNode );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", OnnxNode );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", OnnxNode );
}

//---------------------------------------------------------------------------------------------------------------------

CSigmoidNode::CSigmoidNode( int nodeIndex, const onnx::NodeProto& sigmoid, int opsetVersion ) :
	CActivationNodeBase( nodeIndex, sigmoid, opsetVersion, AF_Sigmoid )
{
}

void CSigmoidNode::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", OnnxNode );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", OnnxNode );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", OnnxNode );
}

//---------------------------------------------------------------------------------------------------------------------

CTanhNode::CTanhNode( int nodeIndex, const onnx::NodeProto& tanh, int opsetVersion ) :
	CActivationNodeBase( nodeIndex, tanh, opsetVersion, AF_Tanh )
{
}

void CTanhNode::CheckOnnxNode() const
{
	// v1 - original
	// v6 - removed legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", OnnxNode );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", OnnxNode );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", OnnxNode );
}

} // namespace NeoOnnx
