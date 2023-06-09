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

#include <NeoML/TraditionalML/SubwordEncoderTrainer.h>
#include <BytePairEncoderTrainer.h>
#include <Utf8Tools.h>

namespace NeoML {

CSubwordEncoderTrainer::CSubwordEncoderTrainer( int vocabSize, TAlgorithm a, TBorderHandling b, TVocabPruning v ) :
	desiredVocabSize( vocabSize ),
	borderHandling( b ),
	vocabPruning( v )
{
	NeoAssert( vocabSize > 0 );
	// Unigram is on the way
	NeoAssert( a == TAlgorithm::BPE );
}

CSubwordEncoderTrainer::~CSubwordEncoderTrainer()
{
	delete bpeTrainer;
}

void CSubwordEncoderTrainer::SetCharacterCoverage( double value )
{
	NeoAssert( vocabPruning == TVocabPruning::Coverage );
	NeoAssert( value > 0 );
	coverage = value;
}

void CSubwordEncoderTrainer::SetMandatoryChars( const CArray<CString>& chars )
{
	NeoAssert( vocabPruning == TVocabPruning::Coverage );
	NeoAssert( chars.Size() < desiredVocabSize );
	for( const CString& s : chars ) {
		NeoAssert( GetUtf8CharLength( s[0] ) == s.Length() );
	}
	chars.CopyTo( mandatoryTokens );
}

void CSubwordEncoderTrainer::SetUnknownTokenId( int value )
{
	NeoAssert( value >= 0 );
	encoderUnkTokenId = value;
}

CPtr<ISubwordEncoder> CSubwordEncoderTrainer::Train( const CWordDictionary& frequencyDict )
{
	CWordDictionary charDict = ( vocabPruning == TVocabPruning::ByteBPE ) ?
		getAllBytesDictionary() :
		getInitialDictionary( frequencyDict );

	for( const auto& token : mandatoryTokens ) {
		charDict.AddWord( token );
	}
	NeoAssert( charDict.Size() < desiredVocabSize );

	bpeTrainer = new CBpeTrainer( desiredVocabSize, borderHandling, vocabPruning == TVocabPruning::ByteBPE, encoderUnkTokenId );
	return bpeTrainer->Train( frequencyDict, charDict ).Ptr();
}

// returns a dictionary of single letters found in 'trainData'
CWordDictionary CSubwordEncoderTrainer::getInitialDictionary( const CWordDictionary& trainData ) const
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

// returns a dictionary with all possible bytes
CWordDictionary CSubwordEncoderTrainer::getAllBytesDictionary()
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

} // namespace NeoML
