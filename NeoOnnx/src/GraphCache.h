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

#include "Graph.h"
#include "NeoMLLink.h"
#include "Tensor.h"

namespace NeoOnnx {

// Cache of data associated with every node output in graph
template<class T>
class CGraphCache {
public:
	explicit CGraphCache( const CGraph& graph );

	// Returns the data associated with link.OutputIndex'th output of link.NodeIndex'th node
	T& operator[]( const CLink& link );
	const T& operator[]( const CLink& link ) const;

private:
	CArray<CArray<T>> cache;
};

template<class T>
CGraphCache<T>::CGraphCache( const CGraph& graph )
{
	cache.SetSize( graph.NodeCount() );
	for( int i = 0; i < graph.NodeCount(); ++i ) {
		cache[i].SetSize( graph[i]->OutputCount() );
	}
}

template<class T>
T& CGraphCache<T>::operator[]( const CLink& link )
{
	return cache[link.NodeIndex][link.OutputIndex];
}

template<class T>
const T& CGraphCache<T>::operator[]( const CLink& link ) const
{
	return cache[link.NodeIndex][link.OutputIndex];
}

// Iinstantiations used in NeoOnnx
typedef CGraphCache<CTensor> CTensorCache;
typedef CGraphCache<CTensorDim> CDimCache;
typedef CGraphCache<CNeoMLLink> CNeoMLLinkCache;

} // namespace NeoOnnx
