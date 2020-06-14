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

// The initial data consists of a directed acyclic graph with each arc assigned a quality estimate.
// The arcs going out of a node are sorted in descending order by quality of the rest of the path to the ending node.
// The algorithm generates paths from the starting point of the graph to the ending point in order of decreasing quality.

#pragma once

namespace NeoML {

// Graph class represents a directed acyclic graph. It should implement the following interface:

// Gets the arc starting from the node where prevArc ends, with the number arcIndex
// const GraphArc* GetNextArc(const GraphArc* prevArc, int arcIndex) const
// Gets the arc quality
// Quality GetArcQuality(const GraphArc* arc) const
// Gets the quality of the path to the ending node, starting with this arc
// Quality GetSuffixQuality(const GraphArc* arc) const
// Indicates if the arc ends at the ending node
// bool IsFinalArc(const GraphArc* arc) const

// Quality may be represented by a built-in numerical type
// or a dedicated class like CQuality or CComplexQuality
template<class Graph>
class CGraphGenerator {
public:
	typedef typename Graph::GraphArc GraphArc;
	typedef typename Graph::Quality Quality;
	
	explicit CGraphGenerator( const Graph* _graph, const Quality& _zeroQuality, const Quality& _minQuality ) :
		zeroQuality( _zeroQuality ), minQuality( _minQuality ), graph( _graph ) { maxStepsQueueSize = 1000; }

	// Indicates if the algorithm can build the next path
	bool CanGenerateNextPath() const { return builtPaths.Size() == 0 || steps.Size() > 0; }

	// The quality of the next path
	Quality NextPathQuality() const;

	// Builds the next path
	bool GetNextPath( CArray<const GraphArc*>& path );

	// Sets the limit of step queue size
	void SetMaxStepsQueueSize( int _maxStepsQueueSize ) { maxStepsQueueSize = _maxStepsQueueSize; }

private:
	// An element of the generated path
	struct CPathElement {
		const GraphArc* Arc; // the graph arc
		int ArcIndex; // the number of the arc in its starting node
		int PrevElement; // the index of the previous path element
		Quality PrevQuality; // the quality of the path from the starting node of the graph to this path element
	};
	// The generating step
	struct CStep {
		int PathElementIndex; // the number of the path element from which the step starts
		int ArcIndex; // the number of the arc in its starting node
		Quality SumQuality; // the current path quality
	};
	typedef AscendingByMember<CStep, Quality, &CStep::SumQuality> StepQualityAscending;

	const Quality zeroQuality; // the zero quality value
	const Quality minQuality; // the minimum quality value
	// The graph in which paths are generated
	const Graph* graph;
	// The generated paths
	CFastArray<CPathElement, 10> builtPaths;
	// The steps queue in order of increasing quality
	CPriorityQueue<CFastArray<CStep, 10>, StepQualityAscending> steps;
	// the limit of step queue size
	int maxStepsQueueSize;

	// Building process
	bool buildBestPath();
	bool buildNextPath();
	bool buildPath(const CPathElement& initialElement);
	void addNewStep();
	void getPath(int pathIndex, CArray<const GraphArc*>& path) const;
};

// Builds the best quality path; its ending node is builtPaths.Last()
template<class Graph>
inline bool CGraphGenerator<Graph>::buildBestPath()
{
	NeoPresume( builtPaths.Size() == 0 );
	CPathElement initialElement;
	initialElement.Arc = graph->GetNextArc( 0, 0 );
	if( initialElement.Arc == 0 ) {
		return false;
	}
	initialElement.ArcIndex = 0;
	initialElement.PrevElement = NotFound;
	initialElement.PrevQuality = zeroQuality;
	// Building the best quality path
	return buildPath( initialElement );
}

// Builds the best quality path; its ending node is builtPaths.Last()
template<class Graph>
inline bool CGraphGenerator<Graph>::buildNextPath()
{
	// Finding the best quality step and removing it from the step queue
	CStep bestStep;
	NeoAssert( steps.Pop( bestStep ) );
	// Using this step to build a new path element with the arc that is next in quality
	CPathElement element = builtPaths[bestStep.PathElementIndex];
	element.ArcIndex = bestStep.ArcIndex;
	element.Arc = element.PrevElement == NotFound ? graph->GetNextArc( 0, element.ArcIndex ) :
		graph->GetNextArc( builtPaths[element.PrevElement].Arc, element.ArcIndex );
	// Building the rest of the path
	return buildPath( element );
}

template<class Graph>
inline bool CGraphGenerator<Graph>::buildPath( const CPathElement& initialElement )
{
	CPathElement element = initialElement;
	while( true ) {
		builtPaths.Add( element );
		// Creating and adding a new step
		addNewStep();
		const GraphArc* prevArc = element.Arc;
		// Checking if the arc ends in the ending node
		if( graph->IsFinalArc( prevArc ) ) {
			break;
		}
		// Creating the next element
		element.Arc = graph->GetNextArc( prevArc, 0 );
		if( element.Arc == 0 ) {
			return false;
		}
		element.ArcIndex = 0;
		element.PrevElement = builtPaths.Size() - 1;
		element.PrevQuality += graph->GetArcQuality( prevArc );
	}
	return true;
}

template<class Graph>
inline void CGraphGenerator<Graph>::addNewStep()
{
	const CPathElement& element = builtPaths.Last();
	const GraphArc* prevArc = element.PrevElement == NotFound ? 0 : builtPaths[element.PrevElement].Arc;
	int nextIndex = element.ArcIndex + 1;
	const GraphArc* nextArc = graph->GetNextArc( prevArc, nextIndex );
	if( nextArc != 0 && steps.Size() < maxStepsQueueSize ) {
		CStep step;
		step.ArcIndex = nextIndex;
		step.PathElementIndex = builtPaths.Size() - 1;
		step.SumQuality = element.PrevQuality + graph->GetSuffixQuality( nextArc );
		steps.Push( step );
	}
}

template<class Graph>
inline void CGraphGenerator<Graph>::getPath( int pathIndex, CArray<const GraphArc*>& path ) const
{
	path.DeleteAll();
	for( int i = pathIndex; i != NotFound; i = builtPaths[i].PrevElement ) {
		path.InsertAt( builtPaths[i].Arc, 0 );
	}
}

template<class Graph>
inline bool CGraphGenerator<Graph>::GetNextPath( CArray<const GraphArc*>& path )
{
	if( builtPaths.Size() == 0 ) {
		if( !buildBestPath() ) {
			return false;
		}
	} else if( steps.Size() > 0 ) {
		if( !buildNextPath() ) {
			return false;
		}
	} else {
		return false;
	}
	getPath( builtPaths.Size() - 1, path );
	return true;
}

template<class Graph>
inline typename CGraphGenerator<Graph>::Quality CGraphGenerator<Graph>::NextPathQuality() const
{
	if( builtPaths.Size() == 0 ) {
		const GraphArc* firstArc = graph->GetNextArc( 0, 0 );
		return firstArc == 0 ? minQuality : graph->GetSuffixQuality( firstArc );
	} else if( steps.Size() > 0 ) {
		return steps.Peek().SumQuality;
	}
	return minQuality;
}

} // namespace NeoML
