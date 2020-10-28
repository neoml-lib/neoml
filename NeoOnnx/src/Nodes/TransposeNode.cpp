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

#include "TransposeNode.h"
#include "GraphInput.h"
#include "GraphCache.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

// Returns if subseq is a subsequence from the whole sequence
// In terms that order between subseq elements is stored in whole
// Elements of subseq aren't obliged to appear immediately right after another
static bool isSubsequence( const CTensorDim& subseq, const CTensorDim& whole )
{
	if( subseq.IsEmpty() ) {
		return true;
	}

	int prevElemPos = whole.Find( subseq[0] );
	if( prevElemPos == -1 ) {
		return false;
	}

	for( int i = 1; i < subseq.Size(); ++i ) {
		int currElemPos = whole.Find( subseq[i], prevElemPos + 1 );
		if( currElemPos == NotFound ) {
			return false;
		}
		prevElemPos = currElemPos;
	}

	return true;
}

// Returns if current dimension list is subsequence of the channel-first ordering
static bool isChannelFirst( const CTensorDim& dim )
{
	return isSubsequence( dim,
		{ BD_BatchLength, BD_BatchWidth, BD_ListSize, BD_Channels, BD_Depth, BD_Height, BD_Width } );
}

// Returns if current dimension list is subsequence of the channel-last ordering
static bool isChannelLast( const CTensorDim& dim )
{
	return isSubsequence( dim,
		{ BD_BatchLength, BD_BatchWidth, BD_ListSize, BD_Height, BD_Width, BD_Depth, BD_Channels } );
}

// There is a specific case, when Transpose converting input from channel-last ordering to channel-first
// But NeoML using channel-last already and NeoOnnx converts weights to be compatible with channel-last ordering
// In that case we can just ignore this node
static bool canSkipTranspose( const CNode* inputNode, const CTensorDim& inputDim, const CFastArray<int, 8>& perm )
{
	if( dynamic_cast<const CGraphInput*>( inputNode ) == nullptr ) {
		return false;
	}

	CTensorDim outputDim;
	outputDim.SetSize( inputDim.Size() );
	for( int i = 0; i < outputDim.Size(); ++i ) {
		outputDim[i] = inputDim[perm[i]];
	}

	return isChannelLast( inputDim ) && isChannelFirst( outputDim );
}

typedef CFastArray<TBlobDim, 2> CDimensionPair;

// Builds list of dimension swaps to emulate perm
static void buildSwapList( const CTensorDim& inputDim, const CFastArray<int, 8>& perm, CArray<CDimensionPair>& swaps )
{
	CFastArray<int, 8> currPerm;
	perm.CopyTo( currPerm );

	for( int i = 0; i < currPerm.Size(); ++i ) {
		if( currPerm[i] != i ) {
			const int otherIndex = currPerm.Find( i );
			swaps.SetSize( swaps.Size() + 1 );
			swaps.Last().Add( inputDim[i] );
			swaps.Last().Add( inputDim[otherIndex] );
			swap( currPerm[i], currPerm[otherIndex] );
		}
		assert( currPerm[i] == i );
	}
}

//---------------------------------------------------------------------------------------------------------------------

CTransposeNode::CTransposeNode( int nodeIndex, const onnx::NodeProto& transpose, int opsetVersion ) :
	COpNode( nodeIndex, transpose, opsetVersion )
{
	// The differences between versions are in supported data types and legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", transpose );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", transpose );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", transpose );

	Attributes.GetRequiredIntArray( "perm", perm );
}

void CTransposeNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& /* mathEngine */ )
{
	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );

	const CTensorShape& inputShape = tensors[Input[0]].Shape;
	CheckNeoOnnxSupport( inputShape.Size() == perm.Size(), "perm.Size doesn't match with input dimensions", OnnxNode );

	CTensorShape& outputShape = tensors[Output[0]].Shape;
	outputShape.SetSize( inputShape.Size() );

	for( int i = 0; i < outputShape.Size(); ++i ) {
		outputShape[i] = inputShape[perm[i]];
	}
}

void CTransposeNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
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

void CTransposeNode::AddLayers( const CGraph& graph, const CTensorCache& /* tensors */, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	if( canSkipTranspose( graph[Input[0]], dims[Input[0]], perm ) ) {
		neoMLLinks[Output[0]] = neoMLLinks[Input[0]];
		return;
	}

	CArray<CDimensionPair> swaps;
	buildSwapList( dims[Input[0]], perm, swaps );

	CNeoMLLink currLink = neoMLLinks[Input[0]];
	const CString baseName = "NeoMLLayer" + Str( dnn.GetLayerCount() ) + "_";
	for( int i = 0; i < swaps.Size(); ++i ) {
		CPtr<CTransposeLayer> currLayer = new CTransposeLayer( dnn.GetMathEngine() );
		currLayer->SetName( baseName + Str( i ) );
		currLayer->Connect( 0, *currLink.Layer, currLink.OutputIndex );
		dnn.AddLayer( *currLayer );
		currLink.Layer = currLayer;
		currLink.OutputIndex = 0;
	}

	neoMLLinks[Output[0]] = currLink;
}

} // namespace NeoOnnx
