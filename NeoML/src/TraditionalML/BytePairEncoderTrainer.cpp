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
#include <Utf8Tools.h>

namespace NeoML {
// Start-of-Word token for the internal dictionary
static const CString BowTokenStr( "/\xFF" );
// End-of-Word token for the internal dictionary
static const CString EowTokenStr( "\\\xFF" );
// SentencePiece special token
static const CString SpSpaceStr( "\xE2\x96\x81" );

//--------------------

CBytePairEncoderTrainer::CCandidatePair::CCandidatePair( int left, int right ):
	Left( left ), Right( right )
{
	NeoAssert( left != NotFound );
	NeoAssert( right != NotFound );
}

int CBytePairEncoderTrainer::CCandidatePair::HashKey() const
{
	int hashKey = CDefaultHash<int>::HashKey( Left );
	AddToHashKey( CDefaultHash<int>::HashKey( Right ), hashKey );
	return hashKey;
}

bool CBytePairEncoderTrainer::CCandidatePair::operator==( const CCandidatePair& other ) const
{
	return Left == other.Left && Right == other.Right;
}

//--------------------

CBytePairEncoderTrainer::CCandidateDataComparator::CCandidateDataComparator( bool inQueue ) :
	inQueue( inQueue )
{}

bool CBytePairEncoderTrainer::CCandidateDataComparator::Predicate( const CCandidateData* first,
	const CCandidateData* second ) const
{
	// queue is MAX-heap, therefore all predicates are inverted
	NeoPresume( first != nullptr && second != nullptr );
	NeoPresume( first != second );

	// > by count
	if( inQueue ) {
		if( first->QueueCount != second->QueueCount ) {
			return first->QueueCount < second->QueueCount;
		}
	} else {
		if( first->RealCount != second->RealCount ) {
			return first->RealCount > second->RealCount;
		}
	}

	// < by length
	const int firstLength = first->Text.Length();
	const int secondLength = second->Text.Length();
	if( firstLength != secondLength ) {
		return inQueue ? secondLength < firstLength : firstLength < secondLength;
	}

	// < lexicographically
	const int cmpResult = first->Text.CompareSubstr( 0, second->Text, secondLength );
	NeoPresume( cmpResult != 0 );
	return inQueue ? cmpResult > 0 : cmpResult < 0;
}

//--------------------

CBytePairEncoderTrainer::CBytePairEncoderTrainer( int vocabSize, TBorderHandling b, TVocabPruning v ) :
	desiredVocabSize( vocabSize ),
	borderHandling( b ),
	vocabPruning( v ),
	queue( CCandidateDataComparator( true ) )
{
	NeoAssert( vocabSize > 0 );

	unkToken = 0;
	vocabulary.Add( CToken{ "<UNK>", true } );

	switch( borderHandling ) {
		case TBorderHandling::EndOfWord:
			eowToken = vocabulary.Size();
			vocabulary.Add( CToken{ EowTokenStr, false } );
			break;
		case TBorderHandling::BeginOfWord:
			bowToken = vocabulary.Size();
			vocabulary.Add( CToken{ BowTokenStr, false } );
			break;
		case TBorderHandling::SentencePiece:
			bowToken = vocabulary.Size();
			vocabulary.Add( CToken{ SpSpaceStr, false } );
			break;
		case TBorderHandling::None:
			break;
		default:
			NeoAssert( false );
	}
}

CBytePairEncoderTrainer::~CBytePairEncoderTrainer() {}

void CBytePairEncoderTrainer::SetCharacterCoverage( double fraction )
{
	NeoAssert( fraction > 0 );
	coverage = fraction;
}

void CBytePairEncoderTrainer::AddMandatoryChars( const CArray<CString>& chars )
{
	NeoAssert( chars.Size() < desiredVocabSize );
	for( const CString& s : chars ) {
		NeoAssert( GetUtf8CharLength( s[0] ) == s.Length() );
		addCharToken( CToken{ s, false } );
	}
}

void CBytePairEncoderTrainer::addCharToken( CToken&& token )
{
	const int tokenId = vocabulary.Size();
	if( charTokenIndex.Has( token.Text ) ) {
		return;
	}
	charTokenStorage.Add( new CString( token.Text ) );
	charTokenIndex.Add( *charTokenStorage.Last(), tokenId );
	vocabulary.Add( std::move( token ) );
}

void CBytePairEncoderTrainer::SetUnknownTokenId( int value )
{
	NeoAssert( value >= 0 );
	encoderUnkTokenId = value;
}

CPtr<IBytePairEncoder> CBytePairEncoderTrainer::Train( const CWordDictionary& frequencyDict )
{
	vocabulary.SetBufferSize( desiredVocabSize );

	const CWordDictionary charDict = ( vocabPruning == TVocabPruning::ByteBPE ) ?
		getAllBytesDictionary() :
		getInitialDictionary( frequencyDict );
	for( int i = 0; i < charDict.Size(); ++i ) {
		const CString& ch = charDict.GetWord( i );
		addCharToken( CToken{ ch, false } );
	}

	prepareDataset( frequencyDict );

	addAllBigrams();
	enqueueNewCandidates();
	dropCandidates( desiredVocabSize - vocabulary.Size() );

	while( vocabulary.Size() < desiredVocabSize ) {
		if( queue.IsEmpty() ) {
			break;
		}

		// Find the most frequent pair taking into account that a (real) count might be decreased during previous merges.
		auto* bestCandidate = queue.Peek();
		NeoPresume( checkNaive( bestCandidate ) == bestCandidate->RealCount );

		while( bestCandidate->QueueCount != bestCandidate->RealCount ) {
			bestCandidate->QueueCount = bestCandidate->RealCount;
			queue.PopAndPush( bestCandidate );
			bestCandidate = queue.Peek();
		}
		queue.Pop();

		vocabulary.Add( CToken{ bestCandidate->Text, false } );

		updateStatistics( *bestCandidate, vocabulary.Size() - 1 );
		candidates.Delete( bestCandidate->Pair );

		enqueueNewCandidates();

		if( vocabulary.Size() % 1000 == 0 ) {
			dropCandidates( desiredVocabSize - vocabulary.Size() );
		}
	}

	return createEncoder();
}

// returns a frequency dictionary of single letters found in 'trainData'
CWordDictionary CBytePairEncoderTrainer::getInitialDictionary( const CWordDictionary& trainData ) const
{
	CWordDictionary charDictionary;
	for( int i = 0; i < trainData.Size(); ++i ) {
		const CString& word = trainData.GetWord( i );
		const int64_t count = trainData.GetWordUseCount( i );

		for( int curPos = 0; curPos < word.Length(); ) {
			const int charLength = GetUtf8CharLength( word[curPos] );
			NeoAssert( charLength > 0 );
			NeoAssert( curPos + charLength <= word.Length() );
			charDictionary.AddWord( word.Mid( curPos, charLength ), count );
			curPos += charLength;
		}
	}
	// just sort the counter
	charDictionary.Finalize( 1 );

	if( coverage == 1. ) {
		return charDictionary;
	}

	// Collect the most frequent chars to cover desired fraction of the training data
	int i = 0;
	double cumCoverage = 0.;
	for( ; i < charDictionary.Size(); ++i ) {
		cumCoverage += charDictionary.GetWordFrequency( i );
		if( cumCoverage >= coverage ) {
			break;
		}
	}
	NeoAssert( cumCoverage >= coverage );
	charDictionary.RestrictSize( i );
	return charDictionary;
}

CWordDictionary CBytePairEncoderTrainer::getAllBytesDictionary()
{
	CWordDictionary byteVocab;
	// zero-byte cannot be a part of string
	for( int i = 1; i < 256; ++i ) {
		CString token;
		// signed or unsigned, we cover all possible values
		token += static_cast<char>(i);
		byteVocab.AddWord( token );
	}
	return byteVocab;
}

// translates set of CStrings (trainData) to array of Words (dataset)
void CBytePairEncoderTrainer::prepareDataset( const CWordDictionary& trainData )
{
	dataset.SetBufferSize( trainData.Size() );

	// Buffer to avoid creation of millions of strings. Max length of a utf-8 symbol is 4. +1 for zero-byte.
	char charToken[5]{};

	for( int i = 0; i < trainData.Size(); ++i ) {
		const CString& wordStr = trainData.GetWord( i );

		CWordWithCount& word = dataset.Append();
		word.Count = trainData.GetWordUseCount( i );

		word.Text.SetBufferSize( wordStr.Length() + 1 );

		if( borderHandling == TBorderHandling::BeginOfWord ) {
			word.Text.Add( bowToken );
		}

		for( int curPos = 0; curPos < wordStr.Length(); ) {
			const int charLength = ( vocabPruning == TVocabPruning::ByteBPE ) ? 1 : GetUtf8CharLength( wordStr[curPos] );
			NeoAssert( charLength > 0 );
			NeoAssert( curPos + charLength <= wordStr.Length() );

			for( int c = 0; c < charLength; ++c ) {
				charToken[c] = wordStr[curPos + c];
			}
			charToken[charLength] = 0;
			int tokenId = unkToken;
			charTokenIndex.Lookup( charToken, tokenId );
			word.Text.Add( tokenId );

			curPos += charLength;
		}

		if( borderHandling == TBorderHandling::EndOfWord ) {
			word.Text.Add( eowToken );
		}
	}
}

// generates initial statistics for training based on all possible pairs
void CBytePairEncoderTrainer::addAllBigrams()
{
	for( int wordId = 0; wordId < dataset.Size(); ++wordId ) {
		const auto& word = dataset[wordId];
		CCandidatePair prevPair;
		for( int pos = 0; pos < word.Text.Size() - 1; ++pos ) {
			const CCandidatePair pair{ word.Text[pos], word.Text[pos + 1] };
			// Don't count 'AA' twice in 'AAA'
			if( pair == prevPair ) {
				prevPair = {};
			} else {
				prevPair = pair;
				addPair( pair, wordId, word.Count );
			}
		}
	}
}

// adds pair to statistics
void CBytePairEncoderTrainer::addPair( const CCandidatePair& pair, int wordId, int64_t wordCount )
{
	if( vocabulary[pair.Left].IsUnk || vocabulary[pair.Right].IsUnk ) {
		return;
	}
	const auto mpData = candidates.GetFirstPosition( pair );
	CCandidateData* pairData{};
	if( mpData == NotFound ) {
		pairData = &( candidates.CreateValue( pair ) );
		pairData->Pair = pair;
		pairData->Text = mergeText( pair );
		newCandidates.Add( pairData );
	} else {
		pairData = &candidates.GetValue( mpData );
	}
	// Candidate will be inserted in the queue only when all 'addPair's will be finished.
	pairData->RealCount += wordCount;
	pairData->WordOccurrences.GetOrCreateValue( wordId ) += 1;
}

CString CBytePairEncoderTrainer::mergeText( const CCandidatePair& pair ) const
{
	NeoAssert( pair.Left != NotFound && pair.Right != NotFound );
	return vocabulary[pair.Left].Text + vocabulary[pair.Right].Text;
}

// adds pair from statistics
void CBytePairEncoderTrainer::deletePair( const CCandidatePair& pair, int wordId, int64_t wordCount )
{
	const auto mpData = candidates.GetFirstPosition( pair );
	if( mpData == NotFound ) {
		return;
	}
	CCandidateData& pairData = candidates.GetValue( mpData );
	pairData.RealCount -= wordCount;

	const auto mpCount = pairData.WordOccurrences.GetFirstPosition( wordId );
	auto& count = pairData.WordOccurrences.GetValue( mpCount );
	if( count == 1 ) {
		pairData.WordOccurrences.DeleteAt( mpCount );
	} else {
		NeoPresume( count > 1 );
		count -= 1;
	}
}

// All pairs containing any part of the merged token are replaced with pairs containing the new token.
// Frequencies are being updated accordingly.
void CBytePairEncoderTrainer::updateStatistics( const CCandidateData& newTokenData, int newTokenId )
{
	for( auto mp = newTokenData.WordOccurrences.GetFirstPosition();
		mp != NotFound;
		mp = newTokenData.WordOccurrences.GetNextPosition( mp ) ) 
	{
		const int wordId = newTokenData.WordOccurrences.GetKey( mp );
		const int count = newTokenData.WordOccurrences.GetValue( mp );
		updateOneWordStatistics( newTokenData, newTokenId, dataset[wordId].Text, wordId, count );
	}
}

void CBytePairEncoderTrainer::updateOneWordStatistics( const CCandidateData& newTokenData, int newTokenId,
	CArray<int>& word, int wordId, int newTokenCountInThisWord )
{
	const int64_t wordCount = dataset[wordId].Count;
	int events{};
	bool evenSameTokensLeftwards = true;
	for( int i = 0; i < word.Size() - 1; ++i ) {
		// Found the new token. Merging...
		if( word[i] == newTokenData.Pair.Left && word[i + 1] == newTokenData.Pair.Right ) {
			const int mergedLeft = newTokenData.Pair.Left;
			const int mergedRight = newTokenData.Pair.Right;
			// Process left pair
			if( i != 0 ) {
				const int adjacentLeft = word[i - 1];
				// If 'LR' is what we merge:
				// ...X'LR'...: always delete (XL)
				// ...XLL'LR'...: don't delete pair XL(L'L)R' as it is not counted
				// ...XLLL'LR'...: delete XLL(LL)R
				if( adjacentLeft != mergedLeft || evenSameTokensLeftwards ) {
					const CCandidatePair oldLeftPair{ adjacentLeft, mergedLeft };
					deletePair( oldLeftPair, wordId, wordCount );
				} else {
					++events;
				}
				// 'LR' -> N
				// ...X'LR'...: always add pair (XN)
				// ...XNNN'LR'...->...XNNNN...: add (NN)
				// ...XNN'LR'...->...XNNN...: skip this one
				if( adjacentLeft != newTokenId || evenSameTokensLeftwards ) {
					const CCandidatePair newLeftPair{ adjacentLeft, newTokenId };
					addPair( newLeftPair, wordId, wordCount );
				} else {
					++events;
				}
			}
			// Process right pair
			if( i != word.Size() - 2 ) {
				const int adjacentRight = word[i + 2];
				bool oddSameTokensRightwards = false;
				for( int j = i + 2; j < word.Size(); ++j ) {
					if( mergedRight == word[j] ) {
						oddSameTokensRightwards = !oddSameTokensRightwards;
					} else {
						break;
					}
				}
				// ...'LR'XX...: always delete (RX)
				// ...'LR'RX...: delete (RR)
				// ...'LR'RRX... -> 'N'RRX: keep same # of (RR)
				// ...'LR'RRRX... -> 'N'RRRX: delete (RR)
				if( mergedRight != adjacentRight || oddSameTokensRightwards ) {
					const CCandidatePair oldRightPair{ mergedRight, adjacentRight };
					deletePair( oldRightPair, wordId, wordCount );
				} else {
					++events;
				}

				const CCandidatePair newRightPair{ newTokenId, adjacentRight };
				addPair( newRightPair, wordId, wordCount );
			}

			word[i] = newTokenId;
			word.DeleteAt( i + 1 );

			if( --newTokenCountInThisWord == 0 ) {
				break;
			}
		}

		if( i > 0 && word[i - 1] == word[i] ) {
			evenSameTokensLeftwards = !evenSameTokensLeftwards;
		} else {
			evenSameTokensLeftwards = true;
		}
	}
}

// Push new candidates to the queue with their final (for this step) scores
void CBytePairEncoderTrainer::enqueueNewCandidates()
{
	for( auto* newCandidate : newCandidates ) {
		newCandidate->QueueCount = newCandidate->RealCount;
		queue.Push( newCandidate );
	}
	newCandidates.DeleteAll();
}

// Candidate score can only be decreased. It is safe to drop the end of the queue from time to time.
void CBytePairEncoderTrainer::dropCandidates( int desiredSize )
{
	// Almost nothing to delete, waste of time...
	if( candidates.Size() < 4 * desiredSize ) {
		return;
	}
	// We can't use heapsort here since some realCounts may not be equal to heapCounts
	CArray<CCandidateData*> candidatesToSort;
	candidatesToSort.SetBufferSize( candidates.Size() );
	queue.Detach( candidatesToSort );
	// Partial sort could have looked better here...
	CCandidateDataComparator comparator( false );
	candidatesToSort.QuickSort( &comparator );

	for( int i = desiredSize; i < candidatesToSort.Size(); ++i ) {
		candidates.Delete( candidatesToSort[i]->Pair );
	}
	candidatesToSort.DeleteAt( desiredSize, candidatesToSort.Size() - desiredSize );

	for( auto* candidate : candidatesToSort ) {
		candidate->QueueCount = candidate->RealCount;
	}
	queue.Reset();
	queue.Attach( candidatesToSort );
}

CPtr<IBytePairEncoder> CBytePairEncoderTrainer::createEncoder()
{
	CBytePairEncoder::CParams params;
	params.UseRawBytes = vocabPruning == TVocabPruning::ByteBPE;
	params.UnknownTokenId = encoderUnkTokenId;

	switch( borderHandling ) {
		case TBorderHandling::EndOfWord:
			params.EndOfWordToken = EowTokenStr;
			break;
		case TBorderHandling::BeginOfWord:
			params.StartOfWordToken = BowTokenStr;
			break;
		case TBorderHandling::SentencePiece:
		case TBorderHandling::None:
			break;
		default:
			NeoAssert( false );
	}

	CPtr<CBytePairEncoder> encoder = new CBytePairEncoder( params );
	CBytePairEncoder::CBPEDictionary outputVocab;
	outputVocab.SetBufferSize( vocabulary.Size() );
	for( int i = 1; i < vocabulary.Size() && i < desiredVocabSize; ++i ) {
		outputVocab.Add( vocabulary[i].Text );
	}
	encoder->InitializeUnsafe( outputVocab );
	return encoder.Ptr();
}

// debug function calculating statistics using brute force
int64_t CBytePairEncoderTrainer::checkNaive( CCandidateData* pair )
{
	int64_t count = 0;
	CCandidatePair prevPair{};

	for( int w = 0; w < dataset.Size(); ++w ) {
		prevPair = {};
		const auto& word = dataset[w];
		int thisWordCount = 0;

		for( int i = 0; i < word.Text.Size() - 1; ++i ) {
			CCandidatePair textPair{ word.Text[i], word.Text[i + 1] };
			if( textPair == prevPair ) {
				prevPair = {};
			} else {
				prevPair = textPair;
				if( textPair == pair->Pair ) {
					thisWordCount += 1;
				}
			}
		}
		if( pair->WordOccurrences.Has( w ) ) {
			int fastCount = pair->WordOccurrences[w];
			NeoAssert( fastCount == thisWordCount );
		} else {
			NeoAssert( 0 == thisWordCount );
		}

		count += thisWordCount * word.Count;
	}
	return count;
}
} // namespace NeoML
