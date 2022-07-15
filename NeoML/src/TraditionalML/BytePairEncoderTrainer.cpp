/* Copyright © 2017-2022 ABBYY Production LLC

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

#include <NeoML/TraditionalML/BytePairEncoderTrainer.h>
#include <BytePairEncoder.h>

namespace NeoML {

// Special tokens.
static const CString StartOfWordTokenInternal( "/\xFF" );
static const CString EndOfWordTokenInternal( "\\\xFF" );

///////////////////////////////////////////////////////////////////////////////

CBytePairEncoderTrainer::CBytePairEncoderTrainer( const CParams& params_, const CWordDictionary& dictionary ) :
	params( params_ ),
	stepsCompletedCount( 0 )
{
	NeoAssert( params.MaxSize > 0 );

	// Reserve 1 for 'Unknown' token.
	params.MaxSize--;

	createTrainData( dictionary );
}

CPtr<IBytePairEncoder> CBytePairEncoderTrainer::Train()
{
	TrainSteps( params.MaxSize );
	return GetEncoder();
}

bool CBytePairEncoderTrainer::TrainSteps( int stepsCount )
{
	NeoAssert( stepsCount > 0 );

	if( stepsCompletedCount == 0 ) {

		buildPairDictionary();

		if( stepsCompletedCount > params.MaxSize ) {
			// If the number of distinct chars in the word dictionary exceeds params.MaxSize,
			// no extra tokens should be introduced.
			tokensDictionary.RestrictSize( params.MaxSize );
			return true;
		}
	}

	const int currentStepsCount = calcCurrentStepsCount( params.MaxSize );

	for( int i = 0; i < currentStepsCount; i++ ) {
		if( !trainSingleStep() ) {
			break;
		}
		stepsCompletedCount++;
	}

	return IsTrainingCompleted();
}

bool CBytePairEncoderTrainer::IsTrainingCompleted() const
{
	// No more pairs of neighbour tokens can be added.
	const bool isNoMergeAvailable = stepsCompletedCount > 0
		&& pairDictionary.Size() == 0;
	// No more steps can be completed.
	const bool isTotalStepsCountAchieved = stepsCompletedCount >= params.MaxSize;

	return isNoMergeAvailable
		|| isTotalStepsCountAchieved;
}

CPtr<IBytePairEncoder> CBytePairEncoderTrainer::GetEncoder() const
{
	CPtr<IBytePairEncoder> encoder = new CBytePairEncoder();
	encoder->LoadDictionary( tokensDictionary, 
		params.UseEndOfWordToken ? EndOfWordTokenInternal : "", 
		params.UseStartOfWordToken ? StartOfWordTokenInternal : "" );
	return encoder;
}

void CBytePairEncoderTrainer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );
	archive.Serialize( stepsCompletedCount );
	trainWords.Serialize( archive );
	trainCounts.Serialize( archive );
	tokensDictionary.Serialize( archive );
	pairDictionary.Serialize( archive );
	archive.Serialize( reverseIndex );
}

// Creates train data for CBpeIterativeTrainer.
void CBytePairEncoderTrainer::createTrainData( const CWordDictionary& dictionary )
{
	trainWords.SetSize( dictionary.Size() );
	trainCounts.SetSize( dictionary.Size() );
	for( int i = 0; i < dictionary.Size(); i++ ) {
		CBytePairEncoder::SplitWordIntoInitialTokens( dictionary.GetWord( i ), 
			params.UseStartOfWordToken ? StartOfWordTokenInternal : "", 
			params.UseEndOfWordToken ? EndOfWordTokenInternal : "", 
			trainWords[i] );
		trainCounts[i] = dictionary.GetWordUseCount( i );
	}
}

// Adds token for each char in dictionary and builds dictionary of token pairs (~ neighbour char pairs).
void CBytePairEncoderTrainer::buildPairDictionary()
{
	for( int i = 0; i < trainWords.Size(); i++ ) {
		const CArray<CString>& tokens = trainWords[i];
		NeoAssert( !tokens.IsEmpty() );

		const long long count = trainCounts[i];
		NeoAssert( count > 0 );

		for( int j = 0; j < tokens.Size() - 1; j++ ) {
			const CString pair = CBytePairEncoder::MergeTokens( tokens[j], tokens[j + 1] );
			pairDictionary.AddWord( pair, count );

			reverseIndex.GetOrCreateValue( pair ).Add( i );
			tokensDictionary.AddWord( tokens[j], count );
		}
		tokensDictionary.AddWord( tokens.Last(), count );
	}

	pairDictionary.Finalize( 1 );
	tokensDictionary.Finalize( 1 );

	stepsCompletedCount += tokensDictionary.Size();
}

// Calculates the number of iterations for the current Run.
// Returned value cannot exceed requestedIterationsCount.
int CBytePairEncoderTrainer::calcCurrentStepsCount( int requestedIterationsCount ) const
{
	NeoAssert( requestedIterationsCount > 0 );
	if( IsTrainingCompleted() ) {
		return 0;
	}
	const int remainingIterationsCount = params.MaxSize - stepsCompletedCount;
	NeoAssert( remainingIterationsCount > 0 );
	return min( requestedIterationsCount, remainingIterationsCount );
}

// Performs one iteration of the algorithm.
// Adds the most frequent pair of neighbour tokens to newPairTokens.
// If no pair of neighbour tokens can be created returns false.
bool CBytePairEncoderTrainer::trainSingleStep()
{
	if( pairDictionary.IsEmpty() ) {
		return false;
	}

	// Selection of the most frequent pair of tokens.
	const CString bestPair = pairDictionary.GetWord( 0 );
	tokensDictionary.AddWord( bestPair, pairDictionary.GetWordUseCount( 0 ) );

	// Request for all words containing bestPair. 
	const CHashTable<int>& wordIdsToChange = reverseIndex.Get( bestPair );

	for( THashTablePosition pos = wordIdsToChange.GetFirstPosition(); pos != NotFound;
		pos = wordIdsToChange.GetNextPosition( pos ) )
	{
		const int id = wordIdsToChange.GetValue( pos );
		const CArray<CString>& oldTokens = trainWords[id];
		const long long count = trainCounts[id];

		// Finding positions in the word where bestPair is located. 
		CArray<int> indexesToMerge;
		bool wasPreviousPairMerged = false;
		for( int j = 0; j < oldTokens.Size() - 1; j++ ) {
			const CString pair = CBytePairEncoder::MergeTokens( oldTokens[j], oldTokens[j + 1] );
			// Decreasing count for current pair.
			pairDictionary.AddWord( pair, -count );
			if( pair == bestPair ) {
				if( !wasPreviousPairMerged ) {
					// Careful processing of overlapping bestPairs.
					indexesToMerge.Add( j );
					wasPreviousPairMerged = true;
					continue;
				}
			} else {
				int wordIndexToDeletePosition = reverseIndex[pair].GetPosition( id );
				if( wordIndexToDeletePosition != NotFound ) {
					reverseIndex[pair].DeleteAt( wordIndexToDeletePosition );
				}
			}
			wasPreviousPairMerged = false;
		}

		NeoAssert( !indexesToMerge.IsEmpty() );

		CArray<CString> newTokens;
		int indexesToMergeIndex = 0;
		for( int j = 0; j < oldTokens.Size(); j++ ) {
			if( indexesToMergeIndex >= indexesToMerge.Size() ||
				j != indexesToMerge[indexesToMergeIndex] )
			{
				newTokens.Add( oldTokens[j] );
			} else {
				newTokens.Add( CBytePairEncoder::MergeTokens( oldTokens[j], oldTokens[j + 1] ) );
				indexesToMergeIndex++;
				j++;
			}

			if( newTokens.Size() > 1 ) {
				// Updating counts for pairs in new word.
				const CString pair = CBytePairEncoder::MergeTokens( newTokens[newTokens.Size() - 2], newTokens.Last() );
				reverseIndex.GetOrCreateValue( pair ).Add( id );
				pairDictionary.AddWord( pair, count );
			}
		}

		newTokens.MoveTo( trainWords[id] );
	}

	reverseIndex.Delete( bestPair );
	NeoAssert( pairDictionary.GetWordUseCount( bestPair ) == 0 );
	// Some pairs may have extincted.
	pairDictionary.Finalize( 1 );

	return true;
}

} // namespace NeoML
