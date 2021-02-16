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

#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/Score.h>

namespace NeoML {

double AccuracyScore( const CArray<CClassificationResult>& classificationResult, const IProblem* problem )
{
	// Check that the number of classification results is equal to the number of data vectors
	NeoAssert( classificationResult.Size() == problem->GetVectorCount() );

	// Calculate the total weight of objects classified correctly
	double score = 0;
	for( int i = 0; i < classificationResult.Size(); ++i ) {
		if( classificationResult[i].PreferredClass == problem->GetClass( i ) ) {
			score += problem->GetVectorWeight( i );
		}
	}

	// Calculates the total weight of all objects (used to normalize the score)
	double weightSum = 0;
	for( int i = 0; i < problem->GetVectorCount(); ++i ) {
		weightSum += problem->GetVectorWeight( i );
	}

	return score / weightSum;
}

//-----------------------------------------------------------------------------------------------------------
// Returns the binary class as an integer value (similar to IProblem::GetBinaryClass( index ))
static int GetBinaryClass( int objectClass )
{
	return ( objectClass != 0 ) ? 1 : -1;
}

double F1Score( const CArray<CClassificationResult>& classificationResult, const IProblem* problem )
{
	// Check that the number of classification results is equal to the number of data vectors
	NeoAssert( classificationResult.Size() == problem->GetVectorCount() );

	// Calculate the number of answers of each type: 
	// true* means the objects classified correctly, false* incorrectly
	// *Positive means the objects that got the +1 label, *Negative the -1 label
	double truePositive = 0;
	double falsePositive = 0;
	double trueNegative = 0;
	double falseNegative = 0;

	for( int i = 0; i < classificationResult.Size(); ++i ) {
		switch( GetBinaryClass( classificationResult[i].PreferredClass ) ) {
			case 1:
				if( GetBinaryClass( problem->GetClass( i ) ) == 1 ) {
					truePositive++;
				} else {
					falsePositive++;
				}
				break;
			case -1:
				if( GetBinaryClass( problem->GetClass( i ) ) == -1 ) {
					trueNegative++;
				} else {
					falseNegative++;
				}
				break;
			default:
				NeoAssert( false );
		}
	}

	// Calculate precision and recall
	double precision = truePositive + falsePositive > 0 ? truePositive / ( truePositive + falsePositive ) : 1;
	double recall = truePositive + falseNegative > 0 ? truePositive / ( truePositive + falseNegative ) : 1;

	// Return the F-measure
	return precision + recall > 0 ? 2 * precision * recall / ( precision + recall ) : 0;
}

} // namespace NeoML
