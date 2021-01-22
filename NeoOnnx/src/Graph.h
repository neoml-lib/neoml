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

#pragma once

namespace NeoOnnx {

// Forward declaration(s)
class CNode;

// Link to NodeIndex'th node OutputIndex'th output
struct CLink
{
	CLink() : NodeIndex( NotFound ), OutputIndex( NotFound ) {}
	CLink( int nodeIndex, int outputIndex ) : NodeIndex( nodeIndex ), OutputIndex( outputIndex ) {}

	int NodeIndex; // Node connected to this input
	int OutputIndex; // Node's output number connected to this input
};

// Graph represented as an array of CNodes in topological order
class CGraph {
public:
	CNode* operator[]( int nodeIndex ) { return nodes[nodeIndex]; }
	const CNode* operator[]( int nodeIndex ) const { return nodes[nodeIndex]; }
	const CNode* operator[]( const CLink& link ) const { return nodes[link.NodeIndex]; }

	void Add( CNode* newNode ) { nodes.Add( newNode ); }

	int NodeCount() const { return nodes.Size(); }

	void SetBufferSize( int nodeCount ) { nodes.SetBufferSize( nodeCount ); }

private:
	CPointerArray<CNode> nodes;
};

} // namespace NeoOnnx
