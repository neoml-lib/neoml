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

#include <NeoML/TraditionalML/VariableMatrix.h>

namespace NeoML {

// The optimal match problem
// The input data is a square matrix with penalties for matching the corresponding elements
// Each element is of Element::Quality type and provides an Element::Quality Element::Penalty() const method
// The penalty of matching to an empty element (for insertions and skips) is stored in MissedLeftElementPairs() and MissedRightElementPairs()

// Allocating the simple generators
template<int BlockSize>
class CGeneratorAllocator : public IMemoryManager {
public:
	virtual void* Alloc( size_t ) { return allocator.Alloc(); }
	virtual void Free( void* ptr ) { allocator.Free( ptr ); }

#ifdef _DEBUG
	virtual void* Alloc( size_t, const char*, int ) { return allocator.Alloc(); }
#endif

	// Sets the tentative number of blocks to be allocated
	// This value is used to set the size of the next allocated page
	void Reserve( int blocksCount ) { allocator.Reserve( blocksCount ); }

private:
	CHashTableAllocator<CurrentMemoryManager, BlockSize> allocator;
};

//------------------------------------------------------------------------------------------------

template<class Element>
class CMatchingGenerator {
public:
	typedef typename Element::Quality Quality;

	CMatchingGenerator( int numberOfLeftElements, int numberOfRightElements,
		const Quality& zeroQuality, const Quality& maxQuality );
	~CMatchingGenerator();
	// Pairings matrix
	CVariableMatrix<Element>& PairMatrix() { return pairMatrix; }
	const CVariableMatrix<Element>& PairMatrix() const { return pairMatrix; }
	// Matching with an empty element
	CArray<Element>& MissedLeftElementPairs() { return missedLeftElementPairs; }
	const CArray<Element>& MissedLeftElementPairs() const { return missedLeftElementPairs; }
	CArray<Element>& MissedRightElementPairs() { return missedRightElementPairs; }
	const CArray<Element>& MissedRightElementPairs() const { return missedRightElementPairs; }
	// Builds the generator
	void Build();
	// Gets the next match; returns false in case of failure
	Quality GetNextMatching( CArray<Element>& matching );
	// Checks if the next match can be generated
	bool CanGenerateNextMatching() const;
	// Returns the lower estimate for the next match penalty
	Quality EstimateNextMatchingPenalty() const;

	// Returns the maximum acceptable match penalty 
	void SetMaxMatchingPenalty( Quality maxMatchingPenalty );

private:
	// The set of graph nodes
	typedef CDynamicBitSet<128> CVertexSet;

	CVariableMatrix<Element> pairMatrix;  // the pairings matrix
	CVariableMatrix<Quality> costMatrix; // the pairings cost matrix
	CArray<Element> missedLeftElementPairs;  // the penalties for skipping the left element
	CArray<Element> missedRightElementPairs; // the penalties for skipping the right element
	Quality totalMissedElementCost; // the total penalty for skipping

	Quality zeroQuality; // the zero quality value
	Quality maxQuality; // the maximum quality value
	Quality maxMatchingPenalty; // the minimum quality of a generated match

	CArray<Quality> leftMinCosts;  // the qualities of the best arcs from the left
	CVariableMatrix<int> sortedRightVertices; // the right nodes sorted for each left node
	CArray<Quality> rightEstimatedPenaltyDelta; // the array with differences between the quality estimates for the best pairing 
											// when the arcs from the current left node to the specified right node are added
	Quality initialEstimatedPenalty; // the initial estimate for the best pairing
	class VertexCostAscending; // the class to sort graph nodes in order of increasing cost

	int maxNumberOfActiveGenerators; // the maximum number of active generators (that are making a step at the same time)
	int numberOfActiveGenerators; // the current number of active generators
	CArray<int> numberOfGenerators; // the number of generators built; the index in the array is the number of the left node
	int minActiveLeftNode; // the minimum number of the left node for active generators

	class CNextMatchingGenerator;
	using EstimatedPenaltyDescending = DescendingPtrByMethod<CNextMatchingGenerator, Quality,
		&CNextMatchingGenerator::EstimatedPenalty>;
	using CurrentPenaltyDescending = DescendingPtrByMethod<CNextMatchingGenerator, Quality,
		&CNextMatchingGenerator::CurrentPenalty>;
	CPriorityQueue<CFastArray<CNextMatchingGenerator*, 1>, EstimatedPenaltyDescending> nextMatchingGenerators;
	CPriorityQueue<CFastArray<CNextMatchingGenerator*, 1>, CurrentPenaltyDescending> nextMatchings;
	CArray<CNextMatchingGenerator*> usedGenerators; // the used generators (to be deleted)
	
	// Allocating the simple generators
	CGeneratorAllocator<sizeof(CNextMatchingGenerator)>* generatorAllocator;

	Quality getMatching( const CArray<int>& backwardArcs, CArray<Element>& matching ) const;
	void buildNextMatchings();

	Quality calcEstimatedPenaltyDelta( int leftVertex, int rightVertex ) const;
	Quality calcCurrentPenaltyDelta( int leftVertex, int rightVertex ) const;
	void buildSortedRightVertices( int leftVertex );
};

//------------------------------------------------------------------------------------------------

// A simple generator of a partial match (as a point in the state space)
template<class Element>
class CMatchingGenerator<Element>::CNextMatchingGenerator {
public:
	CNextMatchingGenerator( CMatchingGenerator* owner );
	~CNextMatchingGenerator() {}

	// Makes one generator step to the next left node
	CNextMatchingGenerator* StepLeft() const;
	// Makes one generator step to the next right node
	CNextMatchingGenerator* StepRight() const;
	// Gets the upper estimate for the penalty of the best pairing this generator can find
	Quality EstimatedPenalty() const { return estimatedPenalty; }
	// Gets the estimate for the current pairing quality
	Quality CurrentPenalty() const { return currentPenalty; }
	// The current left node
	int LeftVertex() const { return leftVertex; }
	// The current right node: rightVertex > owner->costMatrix.Size() - an empty element
	int RightVertex() const { return rightVertex; }
	// Gets the current pairing for this generator
	void GetCurrentMatching( CArray<int>& backwardArcs ) const;
	// Checks if the generator has built a complete pairing
	bool IsComplete() const { return leftVertex == owner->costMatrix.SizeX() - 1; }

	void* operator new( size_t size, IMemoryManager* generatorAllocator )
	{
		return generatorAllocator->Alloc( size );
	}
	void operator delete( void* ptr, IMemoryManager* generatorAllocator )
	{
		generatorAllocator->Free( ptr );
	}
	// Delete the generator. Unfortunately placement delete may not be called directly
	void Delete()
	{ 
		IMemoryManager* generatorAllocator = owner->generatorAllocator;
		this->~CNextMatchingGenerator();
		generatorAllocator->Free(this);
	}

private:
	// The main generator
	CMatchingGenerator* owner; 
	// The previous generator
	const CNextMatchingGenerator* previous;
	int rightVertexIndex; // the index of the right node in the sorted nodes matrix
	int leftVertex;	 // the left node
	int rightVertex; // the current right node: rightVertex > owner->costMatrix.Size() - an empty element
	CVertexSet prevRightVertices; // the right nodes of the previous generators
	// The lower estimate for the best pairing cost for this generator
	Quality estimatedPenalty;
	// The estimate of the currently generated pairing cost
	Quality currentPenalty;

	void stepRight();
};

template<class Element>
inline CMatchingGenerator<Element>::CNextMatchingGenerator::CNextMatchingGenerator(CMatchingGenerator* owner) : 
	owner(owner),
	previous(0),
	rightVertexIndex(0),
	leftVertex(NotFound),
	rightVertex(NotFound)	
{
	estimatedPenalty = owner->initialEstimatedPenalty;
	currentPenalty = owner->totalMissedElementCost;
}

template<class Element>
inline void CMatchingGenerator<Element>::CNextMatchingGenerator::GetCurrentMatching(CArray<int>& backwardArcs) const
{
	backwardArcs.DeleteAll();
	backwardArcs.Add(NotFound, owner->costMatrix.SizeY());
	if( CurrentPenalty() >= owner->maxQuality ) {
		return;
	}
	for( const CNextMatchingGenerator* generator = this; generator != 0; generator = generator->previous ) {
		if( generator->rightVertex >= 0 && generator->rightVertex < backwardArcs.Size() ) {
			backwardArcs[generator->rightVertex] = generator->leftVertex;
		}
	}
}

template<class Element>
inline typename CMatchingGenerator<Element>::CNextMatchingGenerator* 
	CMatchingGenerator<Element>::CNextMatchingGenerator::StepLeft() const
{
	CNextMatchingGenerator* next = new(owner->generatorAllocator) CNextMatchingGenerator(*this);
	next->previous = this;
	next->leftVertex = leftVertex == NotFound ? 0 : leftVertex + 1;
	next->rightVertexIndex = -1;
	next->rightVertex = NotFound;
	if( rightVertex >= 0 && rightVertex < owner->costMatrix.SizeY() ) {
		next->prevRightVertices.Set(rightVertex);
	}
	next->stepRight();
	return next;
}

template<class Element>
inline typename CMatchingGenerator<Element>::CNextMatchingGenerator* 
	CMatchingGenerator<Element>::CNextMatchingGenerator::StepRight() const
{
	CNextMatchingGenerator* next = new(owner->generatorAllocator) CNextMatchingGenerator(*this);
	next->stepRight();
	return next;
}

// Finds the next right element
template<class Element>
inline void CMatchingGenerator<Element>::CNextMatchingGenerator::stepRight()
{
	if( leftVertex < 0 || leftVertex >= owner->costMatrix.SizeX() ) {
		estimatedPenalty = owner->maxQuality;
		currentPenalty = owner->maxQuality;
		return;
	}
	if( leftVertex >= owner->sortedRightVertices.SizeX() ) {
		owner->buildSortedRightVertices(leftVertex);
	}
	if( rightVertex != NotFound ) {
		estimatedPenalty -= owner->calcEstimatedPenaltyDelta(leftVertex, rightVertex);
		currentPenalty -= owner->calcCurrentPenaltyDelta(leftVertex, rightVertex);
	}
	for( rightVertexIndex++; rightVertexIndex < owner->sortedRightVertices.SizeY(); rightVertexIndex++ ) {
		rightVertex = owner->sortedRightVertices(leftVertex, rightVertexIndex);
		if(!prevRightVertices.Has(rightVertex)) {
			break;
		}
	}
	if( rightVertexIndex < owner->sortedRightVertices.SizeY() ) {
		estimatedPenalty += owner->calcEstimatedPenaltyDelta(leftVertex, rightVertex);
		currentPenalty += owner->calcCurrentPenaltyDelta(leftVertex, rightVertex);
	} else {
		estimatedPenalty = owner->maxQuality; // this means all right nodes were tried
		currentPenalty = owner->maxQuality;
	}
	NeoAssert( estimatedPenalty <= currentPenalty );
}

//-------------------------------------------------------------------------------------------------------------

// Sorting the graph nodes in order of increasing cost
template<class Element>
class CMatchingGenerator<Element>::VertexCostAscending {
public:
	VertexCostAscending( const CArray<Quality>& vertexCosts ) : vertexCosts(vertexCosts) {}
	bool Predicate( const int& first, const int& second ) const { return vertexCosts[first] < vertexCosts[second]; }
	bool IsEqual( const int& first, const int& second ) const { return first == second; }
	void Swap( int& first, int& second ) const { swap( first, second ); }
private:
	const CArray<Quality>& vertexCosts;
};

//------------------------------------------------------------------------------------------------

template<class Element>
inline CMatchingGenerator<Element>::CMatchingGenerator(int numberOfLeftElements, int numberOfRightElements,
		const Quality& _zeroQuality, const Quality& _maxQuality ) :
	pairMatrix(numberOfLeftElements, numberOfRightElements),
	zeroQuality(_zeroQuality),
	maxQuality(_maxQuality),
	maxMatchingPenalty(_maxQuality - 1),
	maxNumberOfActiveGenerators(0),
	numberOfActiveGenerators(0),
	minActiveLeftNode(0)
{
	missedLeftElementPairs.SetSize(numberOfLeftElements);
	missedRightElementPairs.SetSize(numberOfRightElements);

	generatorAllocator = FINE_DEBUG_NEW CGeneratorAllocator<sizeof(CNextMatchingGenerator)>();
}

template<class Element>
inline CMatchingGenerator<Element>::~CMatchingGenerator()
{
	const CFastArray<CNextMatchingGenerator*, 1>& nextMatchingGeneratorsBuffer = nextMatchingGenerators.GetBuffer();
	for( int i = 0; i < nextMatchingGeneratorsBuffer.Size(); i++ ) {
		nextMatchingGeneratorsBuffer[i]->Delete();
	}

	const CFastArray<CNextMatchingGenerator*, 1>& nextMatchingsBuffer = nextMatchings.GetBuffer();
	for( int i = 0; i < nextMatchingsBuffer.Size(); i++ ) {
		nextMatchingsBuffer[i]->Delete();
	}

	for( int i = 0; i < usedGenerators.Size(); i++ ) {
		usedGenerators[i]->Delete();
	}
	delete generatorAllocator;
}

// Calculates the correction for the best penalty estimate after the leftVertex : rightVertex pairing is added
template<class Element>
inline typename CMatchingGenerator<Element>::Quality CMatchingGenerator<Element>::calcEstimatedPenaltyDelta(
	int leftVertex, int rightVertex) const
{
	if( rightVertex < costMatrix.SizeY() ) {
		return costMatrix(leftVertex, rightVertex) - leftMinCosts[leftVertex];
	} else {
		return -leftMinCosts[leftVertex];
	}
}

// Calculates the correction for the current penalty estimate after the leftVertex : rightVertex pairing is added
template<class Element>
inline typename CMatchingGenerator<Element>::Quality CMatchingGenerator<Element>::calcCurrentPenaltyDelta(
	int leftVertex, int rightVertex) const
{
	if( rightVertex < costMatrix.SizeY() ) {
		return costMatrix(leftVertex, rightVertex);
	} else {
		return zeroQuality;
	}
}

template<class Element>
inline void CMatchingGenerator<Element>::buildSortedRightVertices(int leftVertex)
{
	NeoAssert( leftVertex <= sortedRightVertices.SizeX() );
	if( leftVertex < sortedRightVertices.SizeX() ) {
		return;
	}
	sortedRightVertices.AddColumn();
	int* outgoingArcs = sortedRightVertices.Column(leftVertex);
	rightEstimatedPenaltyDelta.DeleteAll();
	for( int i = 0; i < sortedRightVertices.SizeY(); i++ ) {
		outgoingArcs[i] = i;
		rightEstimatedPenaltyDelta.Add(calcEstimatedPenaltyDelta(leftVertex, i));
	}
	VertexCostAscending comparator(rightEstimatedPenaltyDelta);
	QuickSort( outgoingArcs, sortedRightVertices.SizeY(), &comparator );
}

// Build a pairing using the rest of the graph
template<class Element>
inline typename CMatchingGenerator<Element>::Quality 
	CMatchingGenerator<Element>::getMatching( const CArray<int>& backwardArcs, CArray<Element>& matching ) const
{
	// The penalty for empty element matching
	Quality result = zeroQuality;
	matching.DeleteAll();
	matching.SetBufferSize(costMatrix.SizeX() + costMatrix.SizeY());
	CVertexSet usedVertices;
	if( pairMatrix.SizeX() <= pairMatrix.SizeY() ) {
		for( int rightVertex = 0; rightVertex < backwardArcs.Size(); rightVertex++ ) {
			if( backwardArcs[rightVertex] != NotFound ) {
				const Element& pair = pairMatrix(backwardArcs[rightVertex], rightVertex);
				result += pair.Penalty();
				matching.Add(pair);
				usedVertices.Set(backwardArcs[rightVertex]);
			} else {
				const Element& pair = missedRightElementPairs[rightVertex];
				result += pair.Penalty();
				matching.Add(pair);
			}
		}
		for( int leftVertex = 0; leftVertex < pairMatrix.SizeX(); leftVertex++ ) {
			if( !usedVertices.Has(leftVertex) ) {
				const Element& pair = missedLeftElementPairs[leftVertex];
				result += pair.Penalty();
				matching.Add(pair);
			}
		}
	} else {
		// The matrix is transposed during work with the generator
		for( int leftVertex = 0; leftVertex < backwardArcs.Size(); leftVertex++ ) {
			if( backwardArcs[leftVertex] != NotFound ) {
				const Element& pair = pairMatrix(leftVertex, backwardArcs[leftVertex]);
				result += pair.Penalty();
				matching.Add(pair);
				usedVertices.Set(backwardArcs[leftVertex]);
			} else {
				const Element& pair = missedLeftElementPairs[leftVertex];
				result += pair.Penalty();
				matching.Add(pair);
			}
		}
		for( int rightVertex = 0; rightVertex < pairMatrix.SizeY(); rightVertex++ ) {
			if( !usedVertices.Has(rightVertex) ) {
				const Element& pair = missedRightElementPairs[rightVertex];
				result += pair.Penalty();
				matching.Add(pair);
			}
		}
	}
	return result;
}

// Uses a combination of A* and beam search to find the best pairings.
// On each step, we choose the best generator according to quality estimate (the classic A*).
// However, we additionally limit the number of active generators as in beam search,
// to avoid combinatorial explosion in case of estimates degeneration
template<class Element>
inline void CMatchingGenerator<Element>::buildNextMatchings()
{
	for(;;) {
		// Impossible to generate next matching
		if( nextMatchingGenerators.Size() == 0 ) {
			break;
		}
		Quality minPenalty = maxQuality;
		if( nextMatchings.Size() > 0 ) {
			minPenalty = nextMatchings.Peek()->CurrentPenalty();
		}
		CNextMatchingGenerator* bestGenerator = nextMatchingGenerators.Peek();
		// The best generator cannot build a pairing better than the one already available
		if( bestGenerator->EstimatedPenalty() > min(minPenalty, maxMatchingPenalty) ) {
			break;
		}
		// Delete the best generator from queue
		NeoAssert( nextMatchingGenerators.Pop() );
		if(bestGenerator->LeftVertex() != NotFound) {
			numberOfGenerators[bestGenerator->LeftVertex()] -= 1;
			// The best generator is not among the active ones
			if(bestGenerator->LeftVertex() < minActiveLeftNode) {
				usedGenerators.Add(bestGenerator);
				continue;
			}
			numberOfActiveGenerators -= 1;
		}
		// The next step
		CNextMatchingGenerator* currentGenerator = bestGenerator->StepRight();
		if( currentGenerator->EstimatedPenalty() < maxMatchingPenalty ) {
			nextMatchingGenerators.Push( currentGenerator );
			numberOfGenerators[currentGenerator->LeftVertex()] += 1;
			numberOfActiveGenerators += 1;
		} else {
			currentGenerator->Delete();
		}
		if( bestGenerator->IsComplete() ) {
			// Add the used generator to the results array
			nextMatchings.Push( bestGenerator );
		} else {
			CNextMatchingGenerator* nextGenerator  = bestGenerator->StepLeft();
 			if( nextGenerator->EstimatedPenalty() < maxMatchingPenalty ) {
				nextMatchingGenerators.Push( nextGenerator );
				numberOfGenerators[nextGenerator->LeftVertex()] += 1;
				numberOfActiveGenerators += 1;
			} else {
				nextGenerator->Delete();
			}
			usedGenerators.Add(bestGenerator);
		}
		// Move the active generators range
		if(numberOfActiveGenerators > maxNumberOfActiveGenerators) {
			numberOfActiveGenerators -= numberOfGenerators[minActiveLeftNode];
			minActiveLeftNode += 1;
		}
	}
}

template<class Element>
inline typename CMatchingGenerator<Element>::Quality 
	CMatchingGenerator<Element>::GetNextMatching( CArray<Element>& matching )
{
	matching.DeleteAll();
	buildNextMatchings();
	if( nextMatchings.Size() > 0 ) {
		CNextMatchingGenerator* bestMatching = 0;
		NeoAssert( nextMatchings.Pop( bestMatching ) );
		usedGenerators.Add(bestMatching);
		CArray<int> backwardArcs;
		bestMatching->GetCurrentMatching(backwardArcs);
		Quality result = getMatching(backwardArcs, matching);
		NeoAssert(bestMatching->CurrentPenalty() == result);
		return result;
	}
	return maxQuality;
}

template<class Element>
inline bool CMatchingGenerator<Element>::CanGenerateNextMatching() const
{
	return EstimateNextMatchingPenalty() < maxMatchingPenalty;
}

template<class Element>
inline typename CMatchingGenerator<Element>::Quality 
	CMatchingGenerator<Element>::EstimateNextMatchingPenalty() const
{
	Quality result = maxQuality;
	if( nextMatchingGenerators.Size() > 0 ) {
		result = min(result, nextMatchingGenerators.Peek()->EstimatedPenalty());
	}
	if( nextMatchings.Size() > 0 ) {
		result = min(result, nextMatchings.Peek()->CurrentPenalty());
	}
	return result;
}

template<class Element>
inline void CMatchingGenerator<Element>::SetMaxMatchingPenalty(Quality _maxMatchingPenalty)
{
	maxMatchingPenalty = _maxMatchingPenalty;
}

// Builds a generator
// The connections matrix is transformed into a bipartite flow graph
template<class Element>
inline void CMatchingGenerator<Element>::Build()
{
	totalMissedElementCost = zeroQuality;
	for( int i = 0; i < missedLeftElementPairs.Size(); i++ ) {
		totalMissedElementCost += missedLeftElementPairs[i].Penalty();
	}
	for( int i = 0; i < missedRightElementPairs.Size(); i++ ) {
		totalMissedElementCost += missedRightElementPairs[i].Penalty();
	}
	// Create a pairings cost matrix with SizeX() <= SizeY()
	// (necessary for correct work of next pairings generator)
	if( pairMatrix.SizeX() <= pairMatrix.SizeY() ) {
		costMatrix.SetSize(pairMatrix.SizeX(), pairMatrix.SizeY());
		for( int leftVertex = 0; leftVertex < costMatrix.SizeX(); leftVertex++ ) {
			for( int rightVertex = 0; rightVertex < costMatrix.SizeY(); rightVertex++ ) {
				costMatrix(leftVertex, rightVertex) = pairMatrix(leftVertex, rightVertex).Penalty()
					- missedLeftElementPairs[leftVertex].Penalty()
					- missedRightElementPairs[rightVertex].Penalty();
			}
		}
	} else {
		costMatrix.SetSize(pairMatrix.SizeY(), pairMatrix.SizeX());
		for( int leftVertex = 0; leftVertex < costMatrix.SizeX(); leftVertex++ ) {
			for( int rightVertex = 0; rightVertex < costMatrix.SizeY(); rightVertex++ ) {
				costMatrix(leftVertex, rightVertex) = pairMatrix(rightVertex, leftVertex).Penalty()
					- missedLeftElementPairs[rightVertex].Penalty()
					- missedRightElementPairs[leftVertex].Penalty();
			}
		}
	}
	// Find the minimum arc cost for left and right nodes
	initialEstimatedPenalty = totalMissedElementCost;
	leftMinCosts.SetBufferSize(costMatrix.SizeX());
	for( int leftVertex = 0; leftVertex < costMatrix.SizeX(); leftVertex++ ) {
		Quality leftMinPenalty = maxQuality;
		Quality* outgoingArcs = costMatrix.Column(leftVertex);
		for( int rightVertex = 0; rightVertex < costMatrix.SizeY(); rightVertex++ ) {
			leftMinPenalty = min(leftMinPenalty, outgoingArcs[rightVertex]);
		}
		leftMinCosts.Add(min(leftMinPenalty, zeroQuality)); // if an arc cost is greater than zero, better skip it
		initialEstimatedPenalty += leftMinCosts.Last();
	}
	// Calculate the parameters to generate the next pairings
	sortedRightVertices.SetBufferSize(costMatrix.SizeX(), costMatrix.SizeY() + 1);
	sortedRightVertices.SetSize(0, costMatrix.SizeY() + 1);
	rightEstimatedPenaltyDelta.SetBufferSize(costMatrix.SizeY() + 1);
	// Initialize the active generators range parameters
	maxNumberOfActiveGenerators = max(costMatrix.SizeX() * costMatrix.SizeY(), 10);
	numberOfActiveGenerators = 0;
	numberOfGenerators.Add(0, costMatrix.SizeX());
	minActiveLeftNode = 0;
	// Reserve the memory for generators
	generatorAllocator->Reserve(maxNumberOfActiveGenerators);

	// Create the initial generator
	nextMatchingGenerators.Push( new(generatorAllocator) CNextMatchingGenerator(this) );
}

} // namespace NeoML
