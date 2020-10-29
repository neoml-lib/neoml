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

#include "common.h"
#pragma hdrstop

#include "NodeUtils.h"
#include "Node.h"
#include "Nodes/FlattenNode.h"
#include "Nodes/ReshapeNode.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CPtr<CDnnBlob> RepackWeightIfFlattened( const CNode* node, const CTensorCache& tensors, const CDimCache& dims,
	CDnnBlob* weight )
{
	const CReshapeNode* reshape = dynamic_cast<const CReshapeNode*>( node );
	const CFlattenNode* flatten = dynamic_cast<const CFlattenNode*>( node );
	if( ( reshape == nullptr || tensors[reshape->GetInput( 0 )].Shape.Size() <= 2 ) 
		&& flatten == nullptr )
	{
		return weight;
	}

	const CTensorShape& shape = tensors[node->GetInput( 0 )].Shape;
	const CTensorDim& dim = dims[node->GetInput( 0 )];

	CBlobDesc newWeightDesc( CT_Float );
	for( int dimIndex = 0; dimIndex < shape.Size(); ++dimIndex ) {
		newWeightDesc.SetDimSize( dim[dimIndex], shape[dimIndex] );
	}

	const int hw = newWeightDesc.Height() * newWeightDesc.Width();
	const int depth = newWeightDesc.Depth();
	const int channels = newWeightDesc.Channels();

	if( ( hw == 1 && depth == 1 )
		|| ( hw == 1 && channels == 1 )
		|| ( depth == 1 && channels == 1 ) )
	{
		return weight;
	}

	// Weights needs conversion from CHW to HWC
	IMathEngine& mathEngine = weight->GetMathEngine();
	CPtr<CDnnBlob> newWeight = weight->GetClone();
	mathEngine.TransposeMatrix( weight->GetObjectCount(), weight->GetData(), newWeightDesc.Channels(), newWeightDesc.Depth(),
		newWeightDesc.Height() * newWeightDesc.Width(), 1, newWeight->GetData(), newWeight->GetDataSize() );
	return newWeight;
}

} // namespace NeoOnnx