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

// A generator of fixed-length sequences

// The input data contains a set of arrays with alternative element variants
// The set is sorted in order of decreasing quality
// 
// The generator creates a set of element variants on each step, 
// so that the total quality of the generated set does not increase
//
// The generated variants are presented as a list of "steps," differences of the generated sequence from the optimal one
// A simple step contains the number of alternative hypotheses array and the index of the hypothesis in it
// A complex step is a list of simple steps, implemented using the index of the previous step in executedSteps array
// In addition, there is an array of initial steps (that create a variant next to the optimal) sorted in order of decreasing quality
//
// Performing a step means generating the variant described by this step
// The generator contains two step queues - the steps to be performed and the steps already performed; both are sorted by quality
//
// The generator takes the best step from the steps queue, builds the variant it gives and puts it into the executedSteps queue
// This creates up to 3 new steps (see the processExecutedStep() method description)
//
// This generator is an optimized version of paths generator for a directed acyclic graph

#pragma once

#include <NeoML/NeoMLCommon.h>

namespace NeoML {

// Sorting mechanism for CSimpleGenerator::Element; by default uses Element::VariantQuality(),
// but you may implement specific procedures for different Element types
template<class Element>
int CompareElements( const Element& first, const Element& second )
{
	if( first.VariantQuality() > second.VariantQuality() ) {
		return 1;
	} else if( first.VariantQuality() < second.VariantQuality() ) {
		return -1;
	} else {
		return 0;
	}
}

// Each element should provide the Element::Quality type and the Element::Quality Element::VariantQuality() const method
// Any built-in arithmetic type may be used for Element::Quality
// You can also implement a special class for rational numbers or numbers with fixed decimal point
template<class Element, int MaxVariantsParam = 64>
class CSimpleGenerator {
public:
	static const int MaxVariants = MaxVariantsParam; // a constant for external use = MaxVariantsParam
	typedef typename Element::Quality Quality;
	
	CSimpleGenerator( const Quality& zeroQuality, const Quality& minQuality );

	// The number of elements in the generated sequences
	int NumberOfElements() const { return Variants.Size(); }

	// The number of variants the element has
	int NumberOfVariants( int elementIndex ) const { return Variants[elementIndex].Size(); }

	// Checks if the next sequence can be generated
	bool CanGenerateNextSet() const { return canGenerateNextStep; }

	// The next sequence quality
	Quality NextSetQuality() const;

	// Builds the next sequence
	// Returns false if that is impossible
	bool GetNextSet( CArray<Element>& elementSet );
	
	// Writes the full information about generator state into a logging stream for debugging
	void WriteGeneratorStateToLog( CTextStream& log ) const;

protected:
	CArray<CArray<Element>> Variants; // the set of alternative variants arrays
	// Sets the number of elements with variants to go through
	void SetNumberOfElements( int numberOfElements );

private:
	struct CStep {
		short ElementIndex; // the index of the element
		short VariantIndex; // the index of the variant
		Quality QualityDifference; // the difference in quality between the best and the current variants
		Quality TotalQualityDifference; // the total quality of the complex step
		int PrevStep; // the index of the previous step in executedSteps for complex steps; NotFound for simple steps
		short InitialStepIndex; // the index in the initialSteps array; used to generate next steps

		CStep( const Quality& zeroQuality );
		bool IsNull() const { return VariantIndex == 0; }
	};
	typedef AscendingByMember<CStep, Quality, &CStep::TotalQualityDifference> TotalQualityDifferenceAscending;
	typedef DescendingByMember<CStep, Quality, &CStep::TotalQualityDifference> TotalQualityDifferenceDescending;
	
	const Quality zeroQuality; // the zero quality value
	const Quality minQuality; // the minimum acceptable quality
	CArray<CStep> initialSteps; // the array of starting steps sorted in order or decreasing quality
	// The steps to be performed, sorted in order of increasing quality
	CPriorityQueue<CFastArray<CStep, 10>, TotalQualityDifferenceAscending> steps;
	CArray<CStep> executedSteps; // the steps already performed, sorted in order of decreasing quality
	CStep currentStep; // the step that was used to build the current sequence
	bool bestSetBuilt; // indicates if the best sequence has been achieved
	Quality bestSetQuality; // the best sequence quality
	bool canGenerateNextStep; // indicates if the next step can be generated

	class CVariantsComparer {
	public:
		bool Predicate( const Element& first, const Element& second ) const
			{ return CompareElements( first, second ) >= 0; }
		bool IsEqual( const Element& first, const Element& second ) const
			{ return CompareElements( first, second ) == 0; }
		void Swap( Element& first, Element& second ) const { swap( first, second ); }
	};

	void getLastSetVariantNumbers( CFastArray<int, 32>& variantNumbers ) const;
	bool buildSet( CArray<Element>& elementSet );
	void sortVariants();
	void buildInitialSteps();
	void processExecutedStep( const CStep& step );
	void insertNewStep( const CStep& step );
	void addNextStep_IncVariantIndex( const CStep& step );
	void addNextStep_AddNewInitialStep( const CStep& step );
	void addNextStep_IncInitialStepIndex( const CStep& step );
	void writeStepToLog( CTextStream& log, const CStep& step, const CString& title ) const;
};

//---------------------------------------------------------------------------------------------------------

template<class Element, int MaxVariantsParam>
inline CSimpleGenerator<Element, MaxVariantsParam>::CSimpleGenerator( const Quality& _zeroQuality,
		const Quality& _minQuality ) :
	zeroQuality( _zeroQuality ),
	minQuality( _minQuality ),
	currentStep( _zeroQuality ),
	bestSetBuilt( false ),
	bestSetQuality( _minQuality ),
	canGenerateNextStep( true )	
{
}

template<class Element, int MaxVariantsParam>
inline void CSimpleGenerator<Element, MaxVariantsParam>::SetNumberOfElements( int numberOfElements )
{
	Variants.SetSize( numberOfElements );
}

template<class Element, int MaxVariantsParam>
inline CSimpleGenerator<Element, MaxVariantsParam>::CStep::CStep( const Quality& zeroQuality )
{
	ElementIndex = 0;
	VariantIndex = 0;
	TotalQualityDifference = zeroQuality;
	QualityDifference = zeroQuality;
	PrevStep = NotFound;
	InitialStepIndex = NotFound;
}

// Inserts the specified step into the sorted array of steps to be performed
template<class Element, int MaxVariantsParam>
inline void CSimpleGenerator<Element, MaxVariantsParam>::insertNewStep( const CStep& step )
{
	steps.Push( step );
}

// Builds the sequence from the variants array according to the currentStep
template<class Element, int MaxVariantsParam>
inline bool CSimpleGenerator<Element, MaxVariantsParam>::buildSet( CArray<Element>& elementSet )
{
	elementSet.DeleteAll();
	CFastArray<int, 32> variantIndices;
	getLastSetVariantNumbers(variantIndices);
	for( int i = 0; i < variantIndices.Size(); i++ ) {
		elementSet.InsertAt(Variants[i][variantIndices[i]], i);
	}
	return true;
}

// Generates a new step and adds it to the steps array, by incrementing VariantIndex
template<class Element, int MaxVariantsParam>
inline void CSimpleGenerator<Element, MaxVariantsParam>::addNextStep_IncVariantIndex( const CStep& step )
{
	CStep newStep = step;
	newStep.VariantIndex += 1;
	const CArray<Element>& alternativeVariants = Variants[newStep.ElementIndex];
	if( newStep.VariantIndex >= alternativeVariants.Size() ) {
		return;
	}
	newStep.QualityDifference = alternativeVariants[newStep.VariantIndex].VariantQuality()
		- alternativeVariants[0].VariantQuality();
	newStep.TotalQualityDifference = newStep.QualityDifference;
	if( newStep.PrevStep != NotFound ) {
		newStep.TotalQualityDifference += executedSteps[newStep.PrevStep].TotalQualityDifference;
	}
	insertNewStep( newStep );
}

// Generates a new step and adds it to the steps array, 
// by combining the step with the initial step for the next element in queue.
// This method would not be needed if we generated the complex steps with all initial steps,
// but in that case some of the steps might lead to the same sequence and therefore be redundant
// Sorting the initial steps in order of decreasing quality of the next hypothesis guarantees correct operation
template<class Element, int MaxVariantsParam>
inline void CSimpleGenerator<Element, MaxVariantsParam>::addNextStep_AddNewInitialStep( const CStep& step )
{
	const int newInitialStepIndex = step.InitialStepIndex + 1;
	if( newInitialStepIndex >= initialSteps.Size() ) {
		return;
	}
	CStep newStep = initialSteps[newInitialStepIndex];
	newStep.PrevStep = executedSteps.Size();
	newStep.TotalQualityDifference += step.TotalQualityDifference;
	insertNewStep( newStep );
}

// Generates a new step and adds it to the steps array, by incrementing InitialStepIndex
// (see addNextStep_AddNewInitialStep)
template<class Element, int MaxVariantsParam>
inline void CSimpleGenerator<Element, MaxVariantsParam>::addNextStep_IncInitialStepIndex( const CStep& step )
{
	const int newInitialStepIndex = step.InitialStepIndex + 1;
	// For initial steps only
	if( step.VariantIndex != 1 || newInitialStepIndex >= initialSteps.Size() ) {
		return;
	}
	CStep newStep = initialSteps[newInitialStepIndex];
	newStep.PrevStep = step.PrevStep;
	if( newStep.PrevStep != NotFound ) {
		newStep.TotalQualityDifference += executedSteps[newStep.PrevStep].TotalQualityDifference;
	}
	insertNewStep( newStep );
}

// Process the step just performed
// Up to 3 new steps may be generated
template<class Element, int MaxVariantsParam>
inline void CSimpleGenerator<Element, MaxVariantsParam>::processExecutedStep( const CStep& step )
{
	// Generate the new steps:
	// 1. By increasing VariantIndex
	addNextStep_IncVariantIndex( step );
	// 2. By adding a new initial step
	addNextStep_AddNewInitialStep( step );
	// 3. By increasing InitialStepIndex
	addNextStep_IncInitialStepIndex( step );
	// Move the step into executedSteps array
	executedSteps.Add( step );
}

template<class Element, int MaxVariantsParam>
inline void CSimpleGenerator<Element, MaxVariantsParam>::sortVariants()
{
	bestSetQuality = zeroQuality;
	for( int i = 0; i < Variants.Size(); i++ ) {
		CArray<Element>& alternativeVariants = Variants[i];
		NeoAssert( alternativeVariants.Size() > 0 );
		// Sort the variants
		alternativeVariants.template QuickSort<CVariantsComparer>();
		bestSetQuality += alternativeVariants[0].VariantQuality();
		// Cut off the variants that are too far along, they will never be reached
		if( alternativeVariants.Size() > MaxVariants ) {
			alternativeVariants.DeleteAt( MaxVariants, alternativeVariants.Size() - MaxVariants );
		}
	}
}

template<class Element, int MaxVariantsParam>
inline bool CSimpleGenerator<Element, MaxVariantsParam>::GetNextSet( CArray<Element>& elementSet )
{
	elementSet.DeleteAll();
	if( !canGenerateNextStep ) {
		return false;
	}

	// Build the best sequence
	if( !bestSetBuilt ) {
		if( bestSetQuality == minQuality ) {
			sortVariants();
		}
		bestSetBuilt = true;
		currentStep = CStep( zeroQuality );
		// Build the initial steps
		buildInitialSteps();
		return buildSet( elementSet );
	}

	// No more possible variants
	if( steps.Size() == 0 ) {
		currentStep = CStep( zeroQuality );
		canGenerateNextStep = false;
		return false;
	}

	// Build the next sequence
	NeoAssert( steps.Pop( currentStep ) );
	// Process the step just performed
	processExecutedStep( currentStep );
	return buildSet( elementSet );
}

template<class Element, int MaxVariantsParam>
inline typename CSimpleGenerator<Element, MaxVariantsParam>::Quality 
	CSimpleGenerator<Element, MaxVariantsParam>::NextSetQuality() const
{
	// The next sequence quality cannot be estimated
	if( !bestSetBuilt ) {
		if( bestSetQuality == minQuality ) {
			const_cast<CSimpleGenerator<Element, MaxVariantsParam>*>(this)->sortVariants();
		}
		return bestSetQuality;
	} else if( steps.Size() > 0 ) {
		return bestSetQuality + steps.Peek().TotalQualityDifference;
	} else {
		return minQuality;
	}
}

// Retrieves the element variants' numbers for the last generated sequence
template<class Element, int MaxVariantsParam>
inline void CSimpleGenerator<Element, MaxVariantsParam>::getLastSetVariantNumbers( CFastArray<int, 32>& variantNumbers ) const
{
	variantNumbers.DeleteAll();
	if( Variants.Size() == 0 ) {
		return;
	}
	variantNumbers.Add( 0, Variants.Size() );
	variantNumbers[currentStep.ElementIndex] = currentStep.VariantIndex;
	for( int i = currentStep.PrevStep; i != NotFound; i = executedSteps[i].PrevStep ) {
		NeoAssert(executedSteps[i].PrevStep < i);
		// The previous step index is always less than the current
		variantNumbers[executedSteps[i].ElementIndex] = executedSteps[i].VariantIndex;
	}
}

// Builds the initial steps
template<class Element, int MaxVariantsParam>
inline void CSimpleGenerator<Element, MaxVariantsParam>::buildInitialSteps()
{
	for( int i = 0; i < Variants.Size(); i++ ) {
		CArray<Element>& alternativeVariants = Variants[i];
		if( Variants[i].Size() > 1 ) {
			CStep step( zeroQuality );
			step.ElementIndex = to<short>(i);
			step.VariantIndex = 1;
			step.QualityDifference = alternativeVariants[1].VariantQuality()
				- alternativeVariants[0].VariantQuality();
			step.TotalQualityDifference = step.QualityDifference;
			initialSteps.Add( step );
		}
	}
	initialSteps.template QuickSort<TotalQualityDifferenceDescending>();
	for( int i = 0; i < initialSteps.Size(); i++ ) {
		initialSteps[i].InitialStepIndex = to<short>(i);
	}
	if( initialSteps.Size() > 0 ) {
		steps.Push( initialSteps.First() );
	}
}

template<class Element, int MaxVariantsParam>
inline void CSimpleGenerator<Element, MaxVariantsParam>::writeStepToLog( CTextStream& log, const CStep& step, const CString& title ) const
{
	if( title == CString() ) {
		log << "Step:\n";
	} else {
		log << title << ":\n";
	}	
	log << "ElementIndex: " << Str( step.ElementIndex ) << "\n";
	log << "VariantIndex: " << Str( step.VariantIndex ) << "\n";
	log << "InitialStepIndex: " << Str( step.InitialStepIndex ) << "\n";
	log << "PrevStep: " << Str( step.PrevStep ) << "\n";
	log << "QualityDifference: " << Str( step.QualityDifference ) << "\n";
	log << "TotalQualityDifference: " << Str( step.TotalQualityDifference ) << "\n";
	log << "\n";
}

// Writes the full information about generator state into a logging stream
template<class Element, int MaxVariantsParam>
inline void CSimpleGenerator<Element, MaxVariantsParam>::WriteGeneratorStateToLog( CTextStream& log ) const
{
	log << "GeneratorState:\n";
	// Variants
	for( int i = 0; i < Variants.Size(); i++ ) {
		log << "Variant " + Str( i ) + ":\n";
		for( int j = 0; j < Variants[i].Size(); j++ ) {
			log << "Element: " << Str( j )
				<< " Quality: " << Str( Variants[i][j].VariantQuality() ) << "\n";
		}
		log << "\n";
	}
	// Initial steps
	log << "InitialSteps:\n";
	for( int i = 0; i < initialSteps.Size(); i++ ) {
		writeStepToLog( log, initialSteps[i], "Step " + Str( i ) );
	}
	log << "\n";
	// Steps to be performed
	log << "Steps to make:\n";
	const CFastArray<CStep, 10>& stepsBuffer = steps.GetBuffer();
	for( int i = 0; i < stepsBuffer.Size(); i++ ) {
		writeStepToLog( log, stepsBuffer[i], "Step " + Str( i ) );
	}
	log << "\n";
	// Steps already performed
	log << "Executed steps:\n";
	for( int i = 0; i < executedSteps.Size(); i++ ) {
		writeStepToLog( log, executedSteps[i], "Step " + Str( i ) );
	}
	log << "\n";
	// The step that was used to generate the current sequence
	writeStepToLog( log, currentStep, "CurrentStep" );
	// Has the best sequence been generated already?
	log << "BestSetBuilt: " << Str( bestSetBuilt ) << "\n";
	// The best sequence quality
	log << "BestSetQuality: " << Str( bestSetQuality ) << "\n";
	// Can the generator create the next step?
	log << "CanGenerateNextStep: " << Str( canGenerateNextStep ) << "\n";
}

} // namespace NeoML
