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

namespace NeoML {

// Linear division graph
template<class Arc>
class CLdGraph {
/*
An arc must provide the InitialCoord() and FinalCoord() methods that return its beginning and end,
the type of quality estimate (Quality) and the Arc::Quality ArcQuality() method that returns the arc quality estimate
class Arc {
public:
	typedef double Quality;
	int InitialCoord() const;
	int FinalCoord() const;
	Quality ArcQuality() const;
};
*/
public:
	typedef typename Arc::Quality Quality;
	
	explicit CLdGraph( int begin = 0 );
	CLdGraph( int begin, int end ); // [begin, end]
	virtual ~CLdGraph();

	// The beginning and ending coordinates
	int Begin() const;
	int End() const;
	// Size
	int Size() const;
	// end should not be less than the current End()
	void ExpandRight( int newEnd );
	// begin should not be greater than the current Begin()
	void ExpandLeft( int newBegin );
	// Resets the LDG: erases all arcs and sets new boundaries
	void Reset( int begin, int end ); // [begin, end]

	// Iterating through the arcs of the "coord" node
	int NumberOfIncomingArcs( int coord ) const;
	int NumberOfOutgoingArcs( int coord ) const;
	const Arc* IncomingArc( int coord, int arcIndex ) const;
	const Arc* OutgoingArc( int coord, int arcIndex ) const;
	Arc* IncomingArc( int coord, int arcIndex );
	Arc* OutgoingArc( int coord, int arcIndex );

	int FindIncomingArc( const Arc* ) const;
	int FindOutgoingArc( const Arc* ) const;
	
	// Checks if an arc with specified coordinates exists
	bool HasArc( int beginCoord, int endCoord ) const;
	// Checks if there is an alternative arc
	bool HasOtherArc( const Arc* arc ) const;
	// Checks if there are arcs with one end inside the specified range and the other outside it
	bool HasArcsCrossingRange( int beginCoord, int endCoord ) const;

	// Inserts a new arc into the linear division graph; the arc object is now owned by the graph object
	// Adds new nodes if necessary
	void InsertArc( Arc* arc );
	// Detaches an arc from the LDG
	// If any nodes are left hanging they are deleted
	void DetachArc( Arc* arc );
	// Detaches an arc and deletes it
	void DeleteArc( Arc* arc );

	// Detaches all arcs and deletes the nodes
	void DetachAll();
	// Deletes all arcs and nodes
	void DeleteAll();

	void DeleteOutgoingArcs( int coord );
	void DeleteIncomingArcs( int coord );
	// Deletes the arcs with specified coordinates
	void DeleteArcs( int beginCoord, int endCoord );
	// Deletes the arcs that start or end in the specified range 
	void DeleteArcsInRange( int from, int to );

	// Deletes the arcs and then recursively removes the hanging nodes
	void DeleteArcRemoveHanging( Arc* arc );
	void DeleteOutgoingArcsRemoveHanging( int coord );
	void DeleteIncomingArcsRemoveHanging( int coord );
	void DeleteArcsRemoveHanging( int beginCoord, int endCoord );
	void DeleteArcsInRangeRemoveHanging( int from, int to );

	// Sorts the arcs in a node
	template <class COMPARE>
	void SortIncomingArcs( int coord, COMPARE* param );
	template <class COMPARE>
	void SortIncomingArcs(int coord);
	template <class COMPARE>
	void SortOutgoingArcs( int coord, COMPARE* param );
	template <class COMPARE>
	void SortOutgoingArcs(int coord);
	template <class COMPARE>
	void InsertInitialCoordOrdered(Arc*, COMPARE* param);
	template <class COMPARE>
	void InsertFinalCoordOrdered(Arc*, COMPARE* param);

	// Checks if there is a graph node at the specified coordinate
	bool HasNode( int coord ) const;
	// Checks if a path connecting two coordinates exists
	bool VerifyPath(int begin, int end) const;
	bool VerifyPath() const { return VerifyPath(Begin(), End()); }

	// Filters out the nodes that cannot be reached from one of the specified coordinates
	void FilterUnreachableNodes( int begin, int end, bool shoudDelete = true );
	void FilterUnreachableNodes() { FilterUnreachableNodes(Begin(), End()); }

	// Calculates the number of arcs crossing the given coordinate
	// (that is, such arcs that arc->InitialCoord() < x and arc->FinalCoord() > x)
	// The ith element contains the data for i + InitialCoord()
	void CalculateNumberOfArcsOverCoord( CArray<int>& buffer ) const;

	// Copies all graph arcs
	void GetAllArcs( CArray<Arc*>& ) const;
	// Copies all arcs that are in full paths from the beginning to the end of the graph
	void GetReachableArcs( CArray<Arc*>& ) const;

	// Calculates the quality of the best existing path (from any point to the end of the graph)
	void CalculateBestPathQuality(Quality minQuality);
	// Calculates the quality of the best path from the given point to the end of the graph
	Quality GetBestPathQuality(int coord) const { return bestPathQuality[coord - begin]; }

// CGraphGenerator methods
	const Arc* GetNextArc( const Arc* prevArc, int arcIndex ) const
	{
		int node = prevArc == 0 ? Begin() : prevArc->FinalCoord();
		if( NumberOfOutgoingArcs( node ) <= arcIndex ) {
			return 0;
		}
		return OutgoingArc( node, arcIndex );
	}
	Quality GetArcQuality(const Arc* arc) const { return arc->ArcQuality(); }
	Quality GetSuffixQuality(const Arc* arc) const { return arc->ArcQuality() + bestPathQuality[arc->FinalCoord() - begin]; }
	bool IsFinalArc(const Arc* arc) const { return ( arc->FinalCoord() == End() ); }

	class SortArcsBySuffixQuality {
	public:
		SortArcsBySuffixQuality(const CLdGraph<Arc>* graph) : graph(graph) {}
		
		bool Predicate( const Arc* first, const Arc* second ) const { return graph->GetSuffixQuality(first) >= graph->GetSuffixQuality(second); }
		bool IsEqual( const Arc* first, const Arc* second  ) const { return graph->GetSuffixQuality(first) == graph->GetSuffixQuality(second); }
		void Swap( Arc*& first, Arc*& second ) const { swap( first, second ); }
	private:
		const CLdGraph<Arc>* graph;
	};

private:
	typedef CFastArray<Arc*, 4> CLdGraphArcArray;
	struct CLdGraphVertex {
		CLdGraphArcArray IncomingArcs;	// incoming arcs for the node
		CLdGraphArcArray OutgoingArcs;	// outgoing arcs for the node
	};
	
	CArray<CLdGraphVertex*> vertices; // the array of all graph nodes; the index of a node is its "begin" coordinate
	int begin;					  // the starting coordinate
	CArray<Quality> bestPathQuality; // the quality of the best path to the end of the graph

	void tryDeleteHangingCoordLeft( CFastArray<int, 10>& coordsToProcess );
	void tryDeleteHangingCoordRight( CFastArray<int, 10>& coordsToProcess );
	// The callback method for additional checks during recursive deletion of arcs
	// The method will be called when the arc has been detached but not yet deleted
	void onCascadeDeleteArc( Arc* ) {}
};

//----------------------------------------------------------------------------------------------------------

template<class Arc>
inline CLdGraph<Arc>::CLdGraph( int begin ) :
	begin(begin)
{
	vertices.Add(0);
}

template<class Arc>
inline CLdGraph<Arc>::CLdGraph( int begin, int end ) :
	begin(begin)
{
	vertices.Add(0, end - begin + 1);
}

template<class Arc>
CLdGraph<Arc>::~CLdGraph()
{
	DeleteAll();
}

template<class Arc>
inline int CLdGraph<Arc>::Begin() const
{
	return begin;
}

template<class Arc>
inline int CLdGraph<Arc>::End() const
{
	return begin + vertices.Size() - 1;
}

template<class Arc>
inline int CLdGraph<Arc>::Size() const
{
	return vertices.Size();
}

template<class Arc>
inline void CLdGraph<Arc>::ExpandRight( int newEnd )
{
	NeoAssert(newEnd >= End());
	vertices.Add(0, newEnd - End());
}

template<class Arc>
inline void CLdGraph<Arc>::ExpandLeft( int newBegin )
{
	NeoAssert(newBegin <= begin);
	int prevEnd = End();
	vertices.InsertAt(0, 0, begin - newBegin);
	begin = newBegin;
	NeoAssert(End() == prevEnd);
}

template<class Arc>
inline void CLdGraph<Arc>::Reset( int newBegin, int newEnd )
{
	NeoAssert(newBegin <= newEnd);
	DeleteAll();
	begin = newBegin;
	int size = newEnd - newBegin + 1;
	if( size < vertices.Size() ) {
		vertices.DeleteAt(size, vertices.Size() - size);
	} else if( size > vertices.Size() ) {
		vertices.Add(0, size - vertices.Size());
	}
}

template<class Arc>
inline int CLdGraph<Arc>::NumberOfIncomingArcs( int coord ) const
{
	int i = coord - begin;
	if( vertices[i] == 0 ) {
		return 0;
	}
	return vertices[i]->IncomingArcs.Size();
}

template<class Arc>
inline int CLdGraph<Arc>::NumberOfOutgoingArcs( int coord ) const
{
	int i = coord - begin;
	if( vertices[i] == 0 ) {
		return 0;
	}
	return vertices[i]->OutgoingArcs.Size();
}

template<class Arc>
inline const Arc* CLdGraph<Arc>::IncomingArc( int coord, int arcIndex ) const
{
	return const_cast<CLdGraph*>(this)->IncomingArc(coord, arcIndex);
}

template<class Arc>
inline const Arc* CLdGraph<Arc>::OutgoingArc( int coord, int arcIndex ) const
{
	return const_cast<CLdGraph*>(this)->OutgoingArc(coord, arcIndex);
}

template<class Arc>
inline Arc* CLdGraph<Arc>::IncomingArc( int coord, int arcIndex )
{
	int i = coord - begin;
	NeoPresume( vertices[i] != 0 );
	return vertices[i]->IncomingArcs[arcIndex];
}

template<class Arc>
inline Arc* CLdGraph<Arc>::OutgoingArc( int coord, int arcIndex )
{
	int i = coord - begin;
	NeoPresume( vertices[i] != 0 );
	return vertices[i]->OutgoingArcs[arcIndex];
}

template<class Arc>
inline bool CLdGraph<Arc>::HasArc( int beginCoord, int endCoord ) const
{
	if( NumberOfOutgoingArcs(beginCoord) == 0 || NumberOfIncomingArcs(endCoord) == 0 ) {
		return false;
	}
	for( int i = 0; i < NumberOfOutgoingArcs(beginCoord); i++ ) {
		if( OutgoingArc(beginCoord, i)->FinalCoord() == endCoord ) {
			return true;
		}
	}
	return false;
}

template<class Arc>
inline bool CLdGraph<Arc>::HasOtherArc( const Arc* arc ) const
{
	for( int i = 0; i < NumberOfOutgoingArcs(arc->InitialCoord()); i++ ) {
		if( OutgoingArc(arc->InitialCoord(), i) != arc
			&& OutgoingArc(arc->InitialCoord(), i)->FinalCoord() == arc->FinalCoord() ) 
		{
			return true;
		}
	}
	return false;
}

template<class Arc> template <class COMPARE>
inline void CLdGraph<Arc>::SortIncomingArcs( int coord, COMPARE* param )
{
	int i = coord - begin;
	if( vertices[i] == 0 ) {
		return;
	}
	vertices[i]->IncomingArcs.QuickSort(param);
}

template<class Arc> template <class COMPARE>
inline void CLdGraph<Arc>::SortIncomingArcs( int coord )
{
	int i = coord - begin;
	if( vertices[i] == 0 ) {
		return;
	}
	vertices[i]->IncomingArcs.template QuickSort<COMPARE>();
}

template<class Arc> template <class COMPARE>
inline void CLdGraph<Arc>::SortOutgoingArcs( int coord, COMPARE* param )
{
	int i = coord - begin;
	if( vertices[i] == 0 ) {
		return;
	}
	vertices[i]->OutgoingArcs.QuickSort(param);
}

template<class Arc> template <class COMPARE>
inline void CLdGraph<Arc>::SortOutgoingArcs( int coord )
{
	int i = coord - begin;
	if( vertices[i] == 0 ) {
		return;
	}
	vertices[i]->OutgoingArcs.template QuickSort<COMPARE>();
}

template<class Arc> template <class COMPARE>
inline void CLdGraph<Arc>::InsertInitialCoordOrdered( Arc* arc, COMPARE* param )
{
	NeoAssert(arc->FinalCoord() > arc->InitialCoord());

	CLdGraphVertex* initial = vertices[arc->InitialCoord() - begin];
	// Create the starting node
	if( initial == 0 ) {
		initial = FINE_DEBUG_NEW CLdGraphVertex();
		vertices[arc->InitialCoord() - begin] = initial;
	}
	int i = initial->OutgoingArcs.FindInsertionPoint(arc, param);
	initial->OutgoingArcs.InsertAt(arc, i);
	// Create the final node
	CLdGraphVertex* final = vertices[arc->FinalCoord() - begin];
	if( final == 0 ) {
		final = FINE_DEBUG_NEW CLdGraphVertex();
		vertices[arc->FinalCoord() - begin] = final;
	}
	final->IncomingArcs.Add(arc);
}

template<class Arc> template <class COMPARE>
inline void CLdGraph<Arc>::InsertFinalCoordOrdered( Arc* arc, COMPARE* param )
{
	NeoAssert(arc->FinalCoord() > arc->InitialCoord());

	CLdGraphVertex* initial = vertices[arc->InitialCoord() - begin];
	// Create the starting node
	if( initial == 0 ) {
		initial = FINE_DEBUG_NEW CLdGraphVertex();
		vertices[arc->InitialCoord() - begin] = initial;
	}
	initial->OutgoingArcs.Add(arc);
	// Create the final node
	CLdGraphVertex* final = vertices[arc->FinalCoord() - begin];
	if( final == 0 ) {
		final = FINE_DEBUG_NEW CLdGraphVertex();
		vertices[arc->FinalCoord() - begin] = final;
	}
	int i = final->IncomingArcs.FindInsertionPoint(arc, param);
	final->IncomingArcs.InsertAt(arc, i);
}

template<class Arc>
inline void CLdGraph<Arc>::InsertArc( Arc* arc )
{
	NeoAssert(arc->FinalCoord() > arc->InitialCoord());

	CLdGraphVertex* initial = vertices[arc->InitialCoord() - begin];
	// Create the starting node
	if( initial == 0 ) {
		initial = FINE_DEBUG_NEW CLdGraphVertex();
		vertices[arc->InitialCoord() - begin] = initial;
	}
	initial->OutgoingArcs.Add(arc);
	// Create the final node
	CLdGraphVertex* final = vertices[arc->FinalCoord() - begin];
	if( final == 0 ) {
		final = FINE_DEBUG_NEW CLdGraphVertex();
		vertices[arc->FinalCoord() - begin] = final;
	}
	final->IncomingArcs.Add(arc);
}	

template<class Arc>
inline void CLdGraph<Arc>::DetachArc( Arc* arc )
{
	// Delete the arcs from the starting node
	CLdGraphVertex* initial = vertices[arc->InitialCoord() - begin];
	NeoPresume( initial != 0 );
	int i = initial->OutgoingArcs.Find(arc);
	NeoAssert( i != NotFound );
	initial->OutgoingArcs.DeleteAt(i);
	// Delete the hanging node
	if( initial->OutgoingArcs.Size() == 0
		&&  initial->IncomingArcs.Size() == 0 )
	{
		delete initial;
		vertices[arc->InitialCoord() - begin] = 0;
	}

	// Delete the arcs to the final node
	CLdGraphVertex* final = vertices[arc->FinalCoord() - begin];
	NeoPresume( final != 0 );
	i = final->IncomingArcs.Find(arc);
	NeoAssert( i != NotFound );
	final->IncomingArcs.DeleteAt(i);
	// Delete the hanging node
	if( final->IncomingArcs.Size() == 0
		&&  final->OutgoingArcs.Size() == 0 )
	{
		delete final;
		vertices[arc->FinalCoord() - begin] = 0;
	}
}

template<class Arc>
inline void CLdGraph<Arc>::DeleteArc( Arc* arc )
{
	DetachArc(arc);
	delete arc;
}

template<class Arc>
inline void CLdGraph<Arc>::DeleteArcRemoveHanging( Arc* arc )
{
	DetachArc(arc);
	CFastArray<int, 10> coordsToProcess;
	coordsToProcess.Add( arc->InitialCoord() );
	tryDeleteHangingCoordLeft( coordsToProcess );
	coordsToProcess.DeleteAll();
	coordsToProcess.Add( arc->FinalCoord() );
	tryDeleteHangingCoordRight( coordsToProcess );
	onCascadeDeleteArc( arc );
	delete arc;
}

// Deletes the hanging nodes (that don't have outgoing arcs) recursively right to left, 
// starting with coordinates listed in coordsToProcess parameter
template<class Arc>
inline void CLdGraph<Arc>::tryDeleteHangingCoordLeft( CFastArray<int, 10>& coordsToProcess )
{
	NeoPresume( !coordsToProcess.IsEmpty() );
	for( int i = 0; i < coordsToProcess.Size(); i++ ) {
		int currCoord = coordsToProcess[i];
		CLdGraphVertex* vertex = vertices[currCoord - begin];
		if( vertex == 0 || !vertex->OutgoingArcs.IsEmpty() ) {
			continue;
		}
		// Delete the incoming arcs
		for( int j = vertex->IncomingArcs.Size() - 1; j >= 0; j-- ) {
			Arc* arc = vertex->IncomingArcs[j];
			vertex->IncomingArcs.DeleteAt( j );

			CLdGraphVertex* initialVertex = vertices[arc->InitialCoord() - begin];
			NeoPresume( initialVertex != 0 );
			int index = initialVertex->OutgoingArcs.Find(arc);
			NeoAssert( index != NotFound );
			initialVertex->OutgoingArcs.DeleteAt(index);
			
			coordsToProcess.Add( arc->InitialCoord() );
			onCascadeDeleteArc( arc );
			delete arc;
		}
		// Delete the node
		delete vertex;
		vertices[currCoord - begin] = 0;
	}
}

// Deletes the hanging nodes (that don't have incoming arcs) recursively left to right
// starting with coordinates listed in coordsToProcess parameter
template<class Arc>
inline void CLdGraph<Arc>::tryDeleteHangingCoordRight( CFastArray<int, 10>& coordsToProcess )
{
	NeoPresume( !coordsToProcess.IsEmpty() );
	for( int i = 0; i < coordsToProcess.Size(); i++ ) {
		int currCoord = coordsToProcess[i];
		CLdGraphVertex* vertex = vertices[currCoord - begin];
		if( vertex == 0 || !vertex->IncomingArcs.IsEmpty() ) {
			continue;
		}
		// Delete the outgoing arcs
		for( int j = vertex->OutgoingArcs.Size() - 1; j >= 0; j-- ) {
			Arc* arc = vertex->OutgoingArcs[j];
			vertex->OutgoingArcs.DeleteAt( j );

			CLdGraphVertex* finalVertex = vertices[arc->FinalCoord() - begin];
			NeoPresume( finalVertex != 0 );
			int index = finalVertex->IncomingArcs.Find(arc);
			NeoAssert( index != NotFound );
			finalVertex->IncomingArcs.DeleteAt(index);
			
			coordsToProcess.Add( arc->FinalCoord() );
			onCascadeDeleteArc( arc );
			delete arc;
		}
		// Delete the node
		delete vertex;
		vertices[currCoord - begin] = 0;
	}
}

template<class Arc>
inline void CLdGraph<Arc>::DeleteAll()
{
	for( int i = 0; i < vertices.Size(); i++ ) {
		if( vertices[i] != 0 ) {
			for( int j = 0; j < vertices[i]->OutgoingArcs.Size(); j++ ) {
				delete vertices[i]->OutgoingArcs[j];
			}
			delete vertices[i];
			vertices[i] = 0;
		}
	}
}

template<class Arc>
inline void CLdGraph<Arc>::DetachAll()
{
	for(int i = 0; i < vertices.Size(); i++) {
		if( vertices[i] != 0 ) {
			delete vertices[i];
			vertices[i] = 0;
		}
	}
}

template<class Arc>
inline bool CLdGraph<Arc>::HasNode( int coord ) const
{
	return NumberOfIncomingArcs(coord) > 0 || NumberOfOutgoingArcs(coord) > 0;
}

template<class Arc>
inline bool CLdGraph<Arc>::VerifyPath( int start, int end ) const
{
	NeoPresume( start >= Begin() && start <= End() 
		&& end >= Begin() && end <= End() 
		&& start <= end);
	
	if( start == end ) {
		return true;
	}
	start -= Begin();
	end -= Begin();
	if( vertices[start] == 0 || vertices[end] == 0 ) {
		return false;
	}
	CDynamicBitSet<> isReachable;
	isReachable.SetBufferSize( vertices.Size() );
	isReachable.Set(start);
	// Traverse the graph from beginning to end, marking the reachable nodes
	for( int i = start; i < end; i++ ) {
		if( vertices[i] != 0 && isReachable[i] ) {
			for( int j = 0; j < vertices[i]->OutgoingArcs.Size(); j++ ) {
				Arc* arc = vertices[i]->OutgoingArcs[j];
				NeoPresume(arc->InitialCoord() - begin == i);
				isReachable.Set(arc->FinalCoord() - begin);
			}
		}
	}
	return isReachable[end];
}

template<class Arc>
inline void CLdGraph<Arc>::FilterUnreachableNodes(int start, int end, bool shouldDelete)
{
	NeoPresume( start >= Begin() && start <= End() 
		&& end >= Begin() && end <= End() );
	
	start -= Begin();
	end -= Begin();

	// Filter out the nodes without incoming arcs
	for( int i = start + 1; i <= end; i++ ) {
		if( vertices[i] != 0 && vertices[i]->IncomingArcs.Size() == 0 ) {
			// Checking that the node isn't hanging
			NeoPresume( vertices[i]->OutgoingArcs.Size() != 0 );
			for( int j = vertices[i]->OutgoingArcs.Size()- 1; j >= 0 ; j-- ) {
				if( shouldDelete ) {
					DeleteArc(vertices[i]->OutgoingArcs[j]);
				} else {
					DetachArc(vertices[i]->OutgoingArcs[j]);
				}
			}
			NeoPresume(vertices[i] == 0);
		}
	}

	// Filter out the nodes without outgoing arcs
	for( int i = end - 1; i >= start; i-- ) {
		if( vertices[i] != 0 && vertices[i]->OutgoingArcs.Size() == 0 ) {
			// Checking that the node isn't hanging
			NeoPresume( vertices[i]->IncomingArcs.Size() != 0 );
			for( int j = vertices[i]->IncomingArcs.Size()- 1; j >= 0 ; j-- ) {
				if( shouldDelete ) {
					DeleteArc(vertices[i]->IncomingArcs[j]);
				} else {
					DetachArc(vertices[i]->IncomingArcs[j]);
				}
			}
			NeoPresume(vertices[i] == 0);
		}
	}
}

template<class T>
inline void CLdGraph<T>::CalculateNumberOfArcsOverCoord( CArray<int>& buffer ) const
{
	buffer.SetSize(vertices.Size());
	for( int i = 0; i < buffer.Size(); i ++ ) {
		buffer[i] = 0;
	}
	// Write the arcs' beginning and ending nodes into an array
	for( int x = Begin(); x <= End(); x++ ) {
		for( int j = 0; j < NumberOfOutgoingArcs(x); j++ ) {
			const T* arc = OutgoingArc(x, j);
			buffer[arc->InitialCoord() - begin + 1] += 1;
			buffer[arc->FinalCoord() - begin] -= 1;
		}
	}
	// Integrate the array; the result in each element is the number of arcs crossing the specified coordinate
	for( int i = 1; i < buffer.Size(); i ++ ) {
		buffer[i] += buffer[i - 1];
		NeoPresume( buffer[i] >= 0 );
	}
}

template<class T>
inline void CLdGraph<T>::DeleteIncomingArcs( int coord )
{
	while( NumberOfIncomingArcs(coord) > 0 ) {
		DeleteArc(IncomingArc(coord, 0));
	}
}

template<class T>
inline void CLdGraph<T>::DeleteIncomingArcsRemoveHanging( int coord )
{
	CFastArray<int, 10> coordsToProcess;
	while( NumberOfIncomingArcs(coord) > 0 ) {
		T* arc = IncomingArc(coord, 0);
		coordsToProcess.Add( arc->InitialCoord() );
		DetachArc( arc );
		onCascadeDeleteArc( arc );
		delete arc;
	}
	if( !coordsToProcess.IsEmpty() ) {
		tryDeleteHangingCoordLeft( coordsToProcess );
	}
	coordsToProcess.DeleteAll();
	coordsToProcess.Add( coord );
	tryDeleteHangingCoordRight( coordsToProcess );
}

template<class T>
inline void CLdGraph<T>::DeleteOutgoingArcs( int coord )
{
	while( NumberOfOutgoingArcs(coord) > 0 ) {
		DeleteArc(OutgoingArc(coord, 0));
	}
}

template<class T>
inline void CLdGraph<T>::DeleteOutgoingArcsRemoveHanging( int coord )
{
	CFastArray<int, 10> coordsToProcess;
	while( NumberOfOutgoingArcs( coord ) > 0 ) {
		T* arc = OutgoingArc( coord, 0 );
		coordsToProcess.Add( arc->FinalCoord() );
		DetachArc( arc );
		onCascadeDeleteArc( arc );
		delete arc;
	}
	if( !coordsToProcess.IsEmpty() ) {
		tryDeleteHangingCoordRight( coordsToProcess );
	}
	coordsToProcess.DeleteAll();
	coordsToProcess.Add( coord );
	tryDeleteHangingCoordLeft( coordsToProcess );
}

template<class T>
inline void CLdGraph<T>::DeleteArcs( int beginCoord, int endCoord )
{
	for( int i = NumberOfOutgoingArcs(beginCoord) - 1; i >= 0; i-- ) {
		if( OutgoingArc(beginCoord, i)->FinalCoord() == endCoord ) {
			DeleteArc(OutgoingArc(beginCoord, i));
		}
	}
}

template<class T>
inline void CLdGraph<T>::DeleteArcsRemoveHanging(int beginCoord, int endCoord)
{
	for( int i = NumberOfOutgoingArcs(beginCoord) - 1; i >= 0; i-- ) {
		T* arc = OutgoingArc( beginCoord, i );
		if( arc->FinalCoord() == endCoord ) {
			DetachArc( arc );
			onCascadeDeleteArc( arc );
			delete arc;
		}
	}
	CFastArray<int, 10> coordsToProcess;
	coordsToProcess.Add( beginCoord );
	tryDeleteHangingCoordLeft( coordsToProcess );
	coordsToProcess.DeleteAll();
	coordsToProcess.Add( endCoord );
	tryDeleteHangingCoordRight( coordsToProcess );
}

template<class T>
inline void CLdGraph<T>::DeleteArcsInRange( int from, int to )
{
	for( int i = from + 1; i < to; i++ ) {
		DeleteOutgoingArcs(i);
		DeleteIncomingArcs(i);
	}
}

template<class T>
inline void CLdGraph<T>::DeleteArcsInRangeRemoveHanging( int from, int to )
{
	CFastArray<int, 10> coordsToProcessLeft, coordsToProcessRight;
	for( int coord = from + 1; coord < to; coord++ ) {
		while( NumberOfOutgoingArcs(coord) > 0 ) {
			T* arc = OutgoingArc(coord, 0);
			if( arc->FinalCoord() >= to ) { // otherwise clearing the arc here
				coordsToProcessRight.Add( arc->FinalCoord() );
			}
			DetachArc( arc );
			onCascadeDeleteArc( arc );
			delete arc;
		}
		while( NumberOfIncomingArcs(coord) > 0 ) {
			T* arc = IncomingArc(coord, 0);
			if( arc->InitialCoord() <= from ) { // otherwise clearing the arc here
				coordsToProcessLeft.Add( arc->InitialCoord() );
			}
			DetachArc( arc );
			onCascadeDeleteArc( arc );
			delete arc;
		}
	}
	if( !coordsToProcessLeft.IsEmpty() ) {
		tryDeleteHangingCoordLeft( coordsToProcessLeft );
	}
	if( !coordsToProcessRight.IsEmpty() ) {
		tryDeleteHangingCoordRight( coordsToProcessRight );
	}
}

template<class T>
inline bool CLdGraph<T>::HasArcsCrossingRange( int beginCoord, int endCoord ) const
{
	for( int i = beginCoord + 1; i < endCoord; i++ ) {
		for( int j = 0; j < NumberOfIncomingArcs(i); j++ ) {
			if( IncomingArc(i, j)->InitialCoord() < beginCoord ) {
				return true;
			}
		}
		for( int j = 0; j < NumberOfOutgoingArcs(i); j++ ) {
			if( OutgoingArc(i, j)->FinalCoord() > endCoord ) {
				return true;
			}
		}
	}
	return false;
}

template<class T>
inline int CLdGraph<T>::FindIncomingArc( const T* arc ) const
{
	for( int j = 0; j < NumberOfIncomingArcs(arc->FinalCoord()); j++ ) {
		if( IncomingArc(arc->FinalCoord(), j) == arc ) {
			return j;
		}
	}
	return NotFound;
}

template<class T>
inline int CLdGraph<T>::FindOutgoingArc( const T* arc ) const
{
	for( int j = 0; j < NumberOfOutgoingArcs(arc->InitialCoord()); j++ ) {
		if( OutgoingArc(arc->InitialCoord(), j) == arc ) {
			return j;
		}
	}
	return NotFound;
}

template<class T>
inline void CLdGraph<T>::GetAllArcs( CArray<T*>& result ) const
{
	result.DeleteAll();
	for(int i = 0; i < vertices.Size(); i++) {
		if( vertices[i] != 0 ) {
			for( int j = 0; j < vertices[i]->OutgoingArcs.Size(); j++ ) {
				result.Add(vertices[i]->OutgoingArcs[j]);
			}
		}
	}
}

template<class T>
inline void CLdGraph<T>::GetReachableArcs( CArray<T*>& result ) const
{
	result.DeleteAll();
	CDynamicBitSet<> isReachable;
	isReachable.SetBufferSize( vertices.Size() );
	isReachable.Set(0);
	for( int i = 0; i < vertices.Size(); i++ ) {
		if( vertices[i] != 0 && isReachable[i] ) {
			for( int j = 0; j < vertices[i]->OutgoingArcs.Size(); j++ ) {
				T* arc = vertices[i]->OutgoingArcs[j];
				isReachable.Set(arc->FinalCoord() - begin);
				result.Add(arc);
			}
		}
	}
}

template<class Arc>
inline void CLdGraph<Arc>::CalculateBestPathQuality(Quality minQuality)
{
	bestPathQuality.DeleteAll();
	bestPathQuality.Add(minQuality, Size());
	bestPathQuality[Size() - 1] -= minQuality;
	// Calculate the best path quality for each point
	for(int x = End() - 1; x >= Begin(); x--) {
		for(int i = 0; i < NumberOfOutgoingArcs(x); i++) {
			bestPathQuality[x - begin] = max(bestPathQuality[x - begin], GetSuffixQuality(OutgoingArc(x, i)));
		}
	}
	SortArcsBySuffixQuality compare(this);
	// Sort the outgoing arcs by the quality of the path to the end of the graph
	for(int x = End() - 1; x >= Begin(); x--) {
		SortOutgoingArcs(x, &compare);		
	}
}

} // namespace NeoML
