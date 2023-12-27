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

#include <UnigramTrainer.h>
#include <Utf8Tools.h>
#include <NeoML/TraditionalML/GraphGenerator.h>

namespace NeoML {
// Start-of-Word token for the internal dictionary
static const CString UnigramBowTokenStr( "/\xFF" );
// End-of-Word token for the internal dictionary
static const CString UnigramEowTokenStr( "\\\xFF" );
// SentencePiece special token
static const CString UnigramSpSpaceStr( "\xE2\x96\x81" );

//----------

CUnigramTrainer::CTrainingSubword::CTrainingSubword( CString text, int64_t count ) :
	CSubword( std::move( text ), 0.0 ),
	Count( count )
{}

//----------

// Element of a trie used to build the initial vocabulary
struct CUnigramTrainer::CTrieCounterData {
	CTrieCounterData() = default;
	CTrieCounterData( int wordId, int begin, int end ) :
		WordId( wordId ), Begin( begin ), End( end )
	{}
	// smth like std::string_view based on the train dataset
	int WordId = NotFound;
	int Begin = NotFound;
	int End = NotFound;
	int64_t Count = 0;
};

//----------

// ~AscendingPtrByMember( Count )
class CUnigramTrainer::CTrieCounterComparator {
public:
	static bool Predicate( const CTrieNode<CTrieCounterData>* first, const CTrieNode<CTrieCounterData>* second )
	{
		return first->Get().Count < second->Get().Count;
	}
	static bool IsEqual( const CTrieNode<CTrieCounterData>* first, const CTrieNode<CTrieCounterData>* second )
	{
		return first->Get().Count == second->Get().Count;
	}
	void Swap( CTrieNode<CTrieCounterData>*& first, CTrieNode<CTrieCounterData>*& second ) const
	{
		swap( first, second );
	}
};

//----------

struct CUnigramTrainer::CTokenLoss {
	explicit CTokenLoss( const CTrainingSubword* token, double loss = 0.0, bool alwaysKeep = false ) :
		Token( token ), Loss( loss ), AlwaysKeep( alwaysKeep )
	{
		NeoAssert( token != nullptr );
	}

	const CTrainingSubword* Token;
	// Penalty for deleting this token
	double Loss;
	// Tokens that have no alternative representations
	bool AlwaysKeep;
};

//----------

CUnigramTrainer::CUnigramTrainer( int vocabSize, TBorderHandling b, bool useByteBpe, int unknownTokenId ) :
	desiredVocabSize( vocabSize )
{
	const bool addBow = b == TBorderHandling::BeginOfWord || b == TBorderHandling::BeginAndEndOfWord;
	params.StartOfWordToken = addBow ? UnigramBowTokenStr
		: ( b == TBorderHandling::SentencePiece ? UnigramSpSpaceStr : "" );
	const bool addEow = b == TBorderHandling::EndOfWord || b == TBorderHandling::BeginAndEndOfWord;
	params.EndOfWordToken = addEow ? UnigramEowTokenStr : "";
	params.UseRawBytes = useByteBpe;
	params.UnknownTokenId = unknownTokenId;

	// Typical vocabSize of modern models is 10-100k, no reason to have 1kk
	NeoAssert( vocabSize < initialVocabSize );
}

CPtr<IUnigramEncoder> CUnigramTrainer::Train( const CWordDictionary& frequencyDict, const CWordDictionary& charVocab )
{
	candidatesTrie.DeleteAll();
	candidatesStorage.DeleteAll();
	chars.DeleteAll();

	chars.SetHashTableSize( charVocab.Size() );
	for( int i = 0; i < charVocab.Size(); ++i ) {
		chars.Add( charVocab.GetWord( i ) );
	}
	fillTrainDict( frequencyDict );
	createInitialVocab();

	while( trainStep() ) {
		continue;
	}

	CArray<IUnigramEncoder::CSubword> resultVocab;
	resultVocab.SetBufferSize( desiredVocabSize );
	dfsTrieToArray( &candidatesTrie, resultVocab );

	addChars( resultVocab );

	CPtr<CUnigramEncoder> encoder = new CUnigramEncoder;
	if( params.EndOfWordToken == UnigramSpSpaceStr ) {
		params.EndOfWordToken.Empty();
	}
	encoder->Initialize( resultVocab, params );
	return encoder.Ptr();
}

int CUnigramTrainer::getTokenLength( const CString& str, int pos ) const
{
	if( params.UseRawBytes ) {
		return 1;
	}

	if( !params.EndOfWordToken.IsEmpty() &&
		str.CompareSubstr( pos, UnigramEowTokenStr, UnigramEowTokenStr.Length() ) == 0 )
	{
		return UnigramEowTokenStr.Length();
	}

	if( !params.StartOfWordToken.IsEmpty() &&
		str.CompareSubstr( pos, params.StartOfWordToken, params.StartOfWordToken.Length() ) == 0 )
	{
		return params.StartOfWordToken.Length();
	}

	const int len = GetUtf8CharLength( str[pos] );
	if( len != 0 ) {
		return len;
	}

	NeoAssert( false );
	return {};
}

// transforms the input dataset adding BoW/EoW and splitting entries by unknown chars
void CUnigramTrainer::fillTrainDict( const CWordDictionary& frequencyDict )
{
	for( int i = 0; i < frequencyDict.Size(); ++i ) {
		const CString& word = frequencyDict.GetWord( i );
		const int64_t count = frequencyDict.GetWordUseCount( i );

		// newWord = <s> + word + </s>
		// if word has <unk>, which doesn't produce subwords, we can consider word as set of independent parts
		CString newWord = params.StartOfWordToken;
		for( int j = 0; j < word.Length(); ) {
			const int charLen = getTokenLength( word, j );
			const CString charStr = word.Mid( j, charLen );
			if( !chars.Has( charStr ) ) {
				// Split the word by unknown chars
				if( newWord.Length() > 0 ) {
					trainDict.AddWord( newWord, count );
					newWord.Empty();
				}
			} else {
				newWord += charStr;
			}

			j += charLen;
		}

		newWord += params.EndOfWordToken;
		if( newWord.Length() > 0 ) {
			trainDict.AddWord( newWord, count );
		}
	}
	trainDict.Finalize( 1 );
}

// Get N most frequent substrings in training data.
// Trie-based solution is suboptimal, but there is no suffix array in NeoML
void CUnigramTrainer::createInitialVocab()
{
	CTrieNode<CTrieCounterData> substringTrie;

	for( int i = 0; i < trainDict.Size(); ++i ) {
		const CString& word = trainDict.GetWord( i );
		const int64_t count = trainDict.GetWordUseCount( i );

		// Add all possible substrings to the trie
		for( int begin = 0; begin < word.Length(); begin += getTokenLength( word, begin ) ) {
			auto* trieNode = &substringTrie;

			int tokenLength = 0;
			for( int end = begin; end < word.Length(); ) {
				// like trie->Add( word[begin:end] ), but without creating substrings
				const int charLen = getTokenLength( word, end );
				for( int l = 0; l < charLen; ++l ) {
					trieNode = trieNode->Add( word[end] );
					++end;
				}

				auto& data = trieNode->Get();
				if( data.WordId == NotFound ) {
					data.WordId = i;
					data.Begin = begin;
					data.End = end;
				}
				data.Count += count;

				if( tokenLength++ == maxSubwordLength ) {
					break;
				}
			}
		}
	}

	// pick initialVocabSize most frequent substrings
	CPriorityQueue<CArray<CTrieNode<CTrieCounterData>*>, CTrieCounterComparator> subwordQueue;
	dfsTrieFillQueue( &substringTrie, subwordQueue );

	for( int i = 0; i < initialVocabSize; ++i ) {
		if( subwordQueue.IsEmpty() ) {
			break;
		}

		const auto& counterData = subwordQueue.Peek()->Get();
		NeoPresume( counterData.WordId != NotFound );

		CString word = trainDict.GetWord( counterData.WordId ).Mid( counterData.Begin,
			counterData.End - counterData.Begin );

		candidatesStorage.Add( new CTrainingSubword{ word, counterData.Count } );
		candidatesTrie.Add( word )->Set( candidatesStorage.Last() );

		subwordQueue.Pop();
	}

	// add chars even if they were not so frequent
	for( auto p = chars.GetFirstPosition(); p != NotFound; p = chars.GetNextPosition( p ) ) {
		const CString& charText = chars.GetValue( p );

		const auto* candidateNode = candidatesTrie.Go( charText );
		if( candidateNode == nullptr || candidateNode->Get() == nullptr ) {
			candidatesStorage.Add( new CTrainingSubword{ charText, 1 } );
			candidatesTrie.Add( charText )->Set( candidatesStorage.Last() );
		}

		const auto* substringNode = substringTrie.Go( charText );
		if( substringNode != nullptr ) {
			candidatesStorage.Last()->Count = substringNode->Get().Count;
		}
	}

	// normalize scores
	int64_t countSum = 0;
	for( int i = 0; i < candidatesStorage.Size(); ++i ) {
		countSum += candidatesStorage[i]->Count;
	}
	for( int i = 0; i < candidatesStorage.Size(); ++i ) {
		candidatesStorage[i]->Score = log( static_cast<double>( candidatesStorage[i]->Count ) / countSum );
	}
}

void CUnigramTrainer::dfsTrieFillQueue( CTrieNode<CTrieCounterData>* node,
	CPriorityQueue<CArray<CTrieNode<CTrieCounterData>*>, CTrieCounterComparator>& outQueue )
{
	if( node == nullptr ) {
		return;
	}

	if( node->Get().Count >= minSubwordFreq ) {
		outQueue.Push( node );
	}

	// no deep recursion (maxSubwordLength), so it's ok
	for( auto p = node->GetFirstChildPos(); p != NotFound; p = node->GetNextChildPos( p ) ) {
		dfsTrieFillQueue( node->GetChild( p ), outQueue );
	}
}

bool CUnigramTrainer::trainStep()
{
	for( int i = 0; i < nEmIterations; ++i ) {
		runEmIteration();
	}

	CArray<CTokenLoss> losses;
	dfsGetLosses( &candidatesTrie, losses );

	using TLossSorter = CompositeComparer<CTokenLoss,
		DescendingByMember<CTokenLoss, bool, &CTokenLoss::AlwaysKeep>,
		DescendingByMember<CTokenLoss, double, &CTokenLoss::Loss>>;
	losses.QuickSort<TLossSorter>();

	int pos = max( desiredVocabSize, static_cast<int>( shrinkingFactor * losses.Size() ) );
	while( pos < losses.Size() && losses[pos].AlwaysKeep ) {
		++pos;
	}

	for( int i = pos; i < losses.Size(); ++i ) {
		candidatesTrie.Go( losses[i].Token->Text )->Set( nullptr );
	}

	// one more step?
	return pos > desiredVocabSize && !losses[pos - 1].AlwaysKeep;
}

void CUnigramTrainer::runEmIteration()
{
	CMap<CString, double> probs;
	for( int i = 0; i < trainDict.Size(); ++i ) {
		calcProbsInWord( trainDict.GetWord( i ), trainDict.GetWordUseCount( i ), probs );
	}
	double sum = 0.0;
	for( auto p = probs.GetFirstPosition(); p != NotFound; p = probs.GetNextPosition( p ) ) {
		sum += probs.GetValue( p );
	}
	for( auto p = probs.GetFirstPosition(); p != NotFound; p = probs.GetNextPosition( p ) ) {
		double& value = probs.GetValue( p );
		NeoPresume( value / sum >= 0.0 );
		NeoPresume( value / sum <= 1.0 );
		value = log( value / sum );
	}

	dfsUpdateTrieProbs( &candidatesTrie, probs );
}

void CUnigramTrainer::calcProbsInWord( const CString& word, int64_t count, CMap<CString, double>& probs ) const
{
	// tokenize
	CPointerArray<CSubwordLdGraphArc> subwordSegments;
	CSubwordLdGraph subwordLdGraph( word );
	FillSubwordLdGraphFromTrie( word, &candidatesTrie, subwordSegments, subwordLdGraph );
	CGraphGenerator<CSubwordLdGraph> graphGen( &subwordLdGraph, 0.0, -FLT_MAX / 2 );

	// get #maxTokenizations best paths and use them to calc scores (probabilities) of used subwords
	double totalPathsProb = 0.0;
	CMap<CString, double> unscaledProbs;
	for( int i = 0; i < maxTokenizations && graphGen.CanGenerateNextPath(); ++i ) {
		CArray<const CSubwordLdGraphArc*> path;
		if( graphGen.NextPathQuality() < minQuality ) {
			break;
		}
		const double pathProb = exp( graphGen.NextPathQuality() );
		graphGen.GetNextPath( path );
		for( const auto& segment : path ) {
			unscaledProbs.GetOrCreateValue( segment->Arc->Text, 0.0 ) += pathProb;
		}
		totalPathsProb += pathProb;
	}

	// scale and add to the output statistics
	for( auto p = unscaledProbs.GetFirstPosition(); p != NotFound; p = unscaledProbs.GetNextPosition( p ) ) {
		const CString& token = unscaledProbs.GetKey( p );
		const double score = static_cast<double>( count ) * unscaledProbs.GetValue( p ) / totalPathsProb;
		probs.GetOrCreateValue( token, 0.0 ) += score;
	}
}

void CUnigramTrainer::dfsUpdateTrieProbs( CTokenTrie* node, const CMap<CString, double>& probs )
{
	if( node == nullptr ) {
		return;
	}

	auto* subword = node->Get();
	if( subword != nullptr ) {
		double proba{};
		if( !probs.Lookup( subword->Text, proba ) ) {
			if( !chars.Has( subword->Text ) ) {
				node->Set( nullptr );
			}
		} else {
			subword->Score = proba;
		}
	}

	for( auto p = node->GetFirstChildPos(); p != NotFound; p = node->GetNextChildPos( p ) ) {
		dfsUpdateTrieProbs( node->GetChild( p ), probs );
	}
}

// Collects losses of all elements of the trie
void CUnigramTrainer::dfsGetLosses( const CTokenTrie* node, CArray<CTokenLoss>& losses ) const
{
	if( node == nullptr ) {
		return;
	}

	const auto* subword = node->Get();
	if( subword != nullptr ) {
		losses.Add( CTokenLoss( subword ) );
		getTokenLoss( subword->Score, subword->Count, losses.Last() );
	}

	for( auto p = node->GetFirstChildPos(); p != NotFound; p = node->GetNextChildPos( p ) ) {
		dfsGetLosses( node->GetChild( p ), losses );
	}
}

// Calc penalty for deleting this token
void CUnigramTrainer::getTokenLoss( double tokenScore, int64_t tokenCount, CTokenLoss& tokenLoss ) const
{
	CPointerArray<CSubwordLdGraphArc> subwordSegments;
	CSubwordLdGraph subwordLdGraph( tokenLoss.Token->Text );
	FillSubwordLdGraphFromTrie( tokenLoss.Token->Text, &candidatesTrie, subwordSegments, subwordLdGraph );
	CGraphGenerator<CSubwordLdGraph> graphGen( &subwordLdGraph, 0.0, -FLT_MAX / 2 );

	NeoAssert( graphGen.CanGenerateNextPath() );
	CArray<const CSubwordLdGraphArc*> path;
	graphGen.GetNextPath( path );

	if( path.Size() != 1 ) {
		// the token can be represented as a concatenation of other tokens with even greater proba. No loss.
		return;
	}

	// the best way to 'tokenize' this token is the token itself (which is quite logical)
	// look at the second best tokenization to see how much worse is it
	NeoAssert( path[0]->Arc->Text == tokenLoss.Token->Text );
	if( graphGen.CanGenerateNextPath() ) {
		const double pathScore = graphGen.NextPathQuality();

		if( graphGen.GetNextPath( path ) ) {
			NeoAssert( path.Size() > 1 );
			tokenLoss.Loss = static_cast<double>( tokenCount ) * ( tokenScore - pathScore );
		} else {
			// yes, sometimes GetNextPath() contradicts with CanGenerateNextPath()
			tokenLoss.AlwaysKeep = true;
		}
	} else {
		// no alternatives, we should keep this token
		tokenLoss.AlwaysKeep = true;
	}
}

// Collects all elements from trie
void CUnigramTrainer::dfsTrieToArray( CTokenTrie* node, CArray<IUnigramEncoder::CSubword>& output )
{
	if( node == nullptr ) {
		return;
	}

	const auto* subword = node->Get();
	if( subword != nullptr ) {
		output.Add( *subword );
	}

	for( auto p = node->GetFirstChildPos(); p != NotFound; p = node->GetNextChildPos( p ) ) {
		dfsTrieToArray( node->GetChild( p ), output );
	}
}

// Adds all chars to the output vocabulary even if they were filtered earlier.
// This is a rare edge case, mostly happens with MandatoryChars defined by user.
void CUnigramTrainer::addChars( CArray<IUnigramEncoder::CSubword>& output ) const
{
	const int sizeBeforeAddingChars = output.Size();

	output.QuickSort<DescendingByMember<IUnigramEncoder::CSubword, double, &IUnigramEncoder::CSubword::Score>>();
	double score = output.Last().Score;
	for( auto p = chars.GetFirstPosition(); p != NotFound; p = chars.GetNextPosition( p ) ) {
		const CString& charText = chars.GetValue( p );
		const auto* candidateNode = candidatesTrie.Go( charText );
		if( candidateNode == nullptr || candidateNode->Get() == nullptr ) {
			score -= scoreEps;
			output.Add( IUnigramEncoder::CSubword{ charText, score } );
		}
	}

	const int overflow = output.Size() - desiredVocabSize;
	if( overflow > 0 ) {
		output.DeleteAt( sizeBeforeAddingChars - 1 - overflow, overflow );
	}
}
} // namespace NeoML
