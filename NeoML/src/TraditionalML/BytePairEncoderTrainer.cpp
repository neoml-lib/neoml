/* Copyright © 2017-2023 ABBYY

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

#include <BytePairEncoderTrainer.h>
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

CBpeTrainer::CCandidatePair::CCandidatePair( int left, int right ) :
	Left( left ),
	Right( right )
{
	NeoAssert( left != NotFound );
	NeoAssert( right != NotFound );
}

int CBpeTrainer::CCandidatePair::HashKey() const
{
	int hashKey = CDefaultHash<int>::HashKey( Left );
	AddToHashKey( CDefaultHash<int>::HashKey( Right ), hashKey );
	return hashKey;
}

bool CBpeTrainer::CCandidatePair::operator==( const CCandidatePair& other ) const
{
	return Left == other.Left && Right == other.Right;
}

//--------------------

bool CBpeTrainer::CCandidateDataComparator::Predicate( const CCandidateData* first,
	const CCandidateData* second )
{
	// queue is MAX-heap, therefore all predicates are inverted
	NeoPresume( first != nullptr && second != nullptr );
	NeoPresume( first != second );

	// > by count
	if( first->QueueCount != second->QueueCount ) {
		return first->QueueCount < second->QueueCount;
	}

	// < by length
	const int firstLength = first->Text.Length();
	const int secondLength = second->Text.Length();
	if( firstLength != secondLength ) {
		return secondLength < firstLength;
	}

	// < lexicographically
	const int cmpResult = first->Text.CompareSubstr( 0, second->Text, secondLength );
	NeoPresume( cmpResult != 0 );
	return cmpResult > 0;
}

//--------------------

CBpeTrainer::CBpeTrainer( int vocabSize, CSubwordEncoderTrainer::TBorderHandling b, bool useByteBpe, int unknownTokenId ) :
	desiredVocabSize( vocabSize ),
	borderHandling( b ),
	useByteBpe( useByteBpe ),
	encoderUnkTokenId( unknownTokenId )
{
	using TBorderHandling = CSubwordEncoderTrainer::TBorderHandling;

	unkToken = 0;
	vocabulary.Add( { "<UNK>", true } );

	switch( borderHandling ) {
		case TBorderHandling::EndOfWord:
			eowToken = vocabulary.Size();
			vocabulary.Add( { EowTokenStr, false } );
			break;
		case TBorderHandling::BeginOfWord:
			bowToken = vocabulary.Size();
			vocabulary.Add( { BowTokenStr, false } );
			break;
		case TBorderHandling::SentencePiece:
			bowToken = vocabulary.Size();
			vocabulary.Add( { SpSpaceStr, false } );
			break;
		case TBorderHandling::BeginAndEndOfWord:
			bowToken = vocabulary.Size();
			vocabulary.Add( { BowTokenStr, false } );
			eowToken = vocabulary.Size();
			vocabulary.Add( { EowTokenStr, false } );
			break;
		case TBorderHandling::None:
			break;
		default:
			NeoAssert( false );
	}
}

CPtr<IBytePairEncoder> CBpeTrainer::Train( const CWordDictionary& frequencyDict, const CWordDictionary& charVocab )
{
	for( int i = 0; i < charVocab.Size(); ++i ) {
		addCharToken( charVocab.GetWord( i ), false );
	}
	prepareDataset( frequencyDict );

	addAllBigrams();
	enqueueNewCandidates();

	while( vocabulary.Size() < desiredVocabSize ) {
		if( queue.IsEmpty() ) {
			break;
		}

		// Find the most frequent pair taking into account that the pair's real count might be decreased during previous merges.
		auto* bestCandidate = queue.Peek();
		NeoPresume( checkNaive( bestCandidate ) == bestCandidate->RealCount );
		if( bestCandidate->QueueCount == 0 ) {
			break;
		}

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
	}

	return createEncoder();
}

// add a single letter to the vocabulary
void CBpeTrainer::addCharToken( const CString& tokenText, bool isUnk )
{
	if( charTokenIndex.Has( tokenText ) ) {
		return;
	}
	const int tokenId = vocabulary.Size();
	charTokenStorage.Add( new CString( tokenText ) );
	charTokenIndex.Add( *charTokenStorage.Last(), tokenId );
	vocabulary.Add( { tokenText, isUnk } );
}

// translate set of CStrings (trainData) to array of Words (dataset)
void CBpeTrainer::prepareDataset( const CWordDictionary& trainData )
{
	using TBorderHandling = CSubwordEncoderTrainer::TBorderHandling;

	dataset.SetBufferSize( trainData.Size() );

	// Buffer to avoid creation of millions of strings. Max length of a utf-8 symbol is 4. +1 for zero-byte.
	char charToken[5]{};

	const bool addBow = borderHandling == TBorderHandling::BeginOfWord ||
		borderHandling == TBorderHandling::SentencePiece ||
		borderHandling == TBorderHandling::BeginAndEndOfWord;
	const bool addEow = borderHandling == TBorderHandling::EndOfWord ||
		borderHandling == TBorderHandling::BeginAndEndOfWord;
	const int auxTokens = static_cast<int>( addBow ) + static_cast<int>( addEow );

	for( int i = 0; i < trainData.Size(); ++i ) {
		const CString& wordStr = trainData.GetWord( i );

		CWordWithCount& word = dataset.Append();
		word.Count = trainData.GetWordUseCount( i );

		word.Text.SetBufferSize( wordStr.Length() + auxTokens );

		if( addBow ) {
			word.Text.Add( bowToken );
		}

		for( int curPos = 0; curPos < wordStr.Length(); ) {
			const int charLength = useByteBpe ? 1 : GetUtf8CharLength( wordStr[curPos] );
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

		if( addEow ) {
			word.Text.Add( eowToken );
		}
	}
}

// generate initial statistics for training based on all possible pairs
void CBpeTrainer::addAllBigrams()
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

// add pair to the statistics
void CBpeTrainer::addPair( const CCandidatePair& pair, int wordId, int64_t wordCount )
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

CString CBpeTrainer::mergeText( const CCandidatePair& pair ) const
{
	NeoAssert( pair.Left != NotFound && pair.Right != NotFound );
	return vocabulary[pair.Left].Text + vocabulary[pair.Right].Text;
}

// subtract pair from the statistics
void CBpeTrainer::deletePair( const CCandidatePair& pair, int wordId, int64_t wordCount )
{
	const auto mpData = candidates.GetFirstPosition( pair );
	NeoAssert( mpData != NotFound );
	CCandidateData& pairData = candidates.GetValue( mpData );
	pairData.RealCount -= wordCount;

	const auto mpCount = pairData.WordOccurrences.GetFirstPosition( wordId );
	auto& count = pairData.WordOccurrences.GetValue( mpCount );
	if( count == 1 ) {
		pairData.WordOccurrences.DeleteAt( mpCount );
		if( pairData.WordOccurrences.IsEmpty() ) {
			NeoAssert( pairData.RealCount == 0 );
			pairData.WordOccurrences.FreeBuffer();
		}
	} else {
		NeoPresume( count > 1 );
		count -= 1;
	}
}

// All pairs containing any part of the merged token are replaced with pairs containing the new token.
// Frequencies are being updated accordingly.
void CBpeTrainer::updateStatistics( const CCandidateData& newTokenData, int newTokenId )
{
	for( auto mp = newTokenData.WordOccurrences.GetFirstPosition();
		mp != NotFound;
		mp = newTokenData.WordOccurrences.GetNextPosition( mp ) ) 
	{
		const int wordId = newTokenData.WordOccurrences.GetKey( mp );
		int count = newTokenData.WordOccurrences.GetValue( mp );
		updateOneWordStatistics( newTokenData, newTokenId, dataset[wordId].Text, wordId, count );
	}
}

// O(len(word)): walk and find new token positions, update counts
// Words are usually not really long, and become shorter during training. Furthermore, CMap is memory-expensive.
void CBpeTrainer::updateOneWordStatistics( const CCandidateData& newTokenData, int newTokenId,
	CArray<int>& word, int wordId, int newTokenCountInThisWord )
{
	const int64_t wordCount = dataset[wordId].Count;
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
				}
				// 'LR' -> N
				// ...X'LR'...: always add pair (XN)
				// ...XNNN'LR'...->...XNNNN...: add (NN)
				// ...XNN'LR'...->...XNNN...: skip this one
				if( adjacentLeft != newTokenId || evenSameTokensLeftwards ) {
					const CCandidatePair newLeftPair{ adjacentLeft, newTokenId };
					addPair( newLeftPair, wordId, wordCount );
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
void CBpeTrainer::enqueueNewCandidates()
{
	for( auto* newCandidate : newCandidates ) {
		newCandidate->QueueCount = newCandidate->RealCount;
		queue.Push( newCandidate );
	}
	newCandidates.DeleteAll();
}

CPtr<IBytePairEncoder> CBpeTrainer::createEncoder()
{
	using TBorderHandling = CSubwordEncoderTrainer::TBorderHandling;

	CBytePairEncoder::CParams params;
	params.UseRawBytes = useByteBpe;
	params.UnknownTokenId = encoderUnkTokenId;

	switch( borderHandling ) {
		case TBorderHandling::EndOfWord:
			params.EndOfWordToken = EowTokenStr;
			break;
		case TBorderHandling::BeginOfWord:
			params.StartOfWordToken = BowTokenStr;
			break;
		case TBorderHandling::BeginAndEndOfWord:
			params.EndOfWordToken = EowTokenStr;
			params.StartOfWordToken = BowTokenStr;
			break;
		case TBorderHandling::SentencePiece:
			// SentencePiece treats space as normal symbol. It should be inserted by user.
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
int64_t CBpeTrainer::checkNaive( CCandidateData* pair )
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
