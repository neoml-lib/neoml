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

#include <BytePairEncoder.h>
#include <SubwordDecoder.h>
#include <Utf8Tools.h>

namespace NeoML {

REGISTER_NEOML_MODEL( CBytePairEncoder, BytePairEncoderModelName )

static const CString UnkToken( "<UNK>" );

void CBytePairEncoder::InitializeUnsafe( const CBPEDictionary& _tokens )
{
	NeoAssert( !IsInitialized() );

	_tokens.CopyTo( tokens );
	tokenToId.SetHashTableSize( tokens.Size() );

	for( int i = 0; i < tokens.Size(); ++i ) {
		const auto& token = tokens[i];
		NeoAssert( !tokenToId.Has( token ) );
		tokenToId.Add( token, tokenToId.Size() );
	}
}

void CBytePairEncoder::Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const
{
	NeoAssert( IsInitialized() );
	if( decoder == nullptr ) {
		CMap<int, CString> idToToken;
		GetIdToTokenMapping( idToToken );
		decoder = std::make_unique<CSubwordDecoder>( params, std::move( idToToken ) );
	}
	decoder->Decode( tokenIds, words );
}

static constexpr int BytePairEncoderImplVersion = 1;
// version 0
static const CString LegacySowToken( "/\xFF" );
static const CString LegacyEowToken( "\\\xFF" );

void CBytePairEncoder::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( BytePairEncoderImplVersion );
	if( version >= 1 ) {
		params.Serialize( archive );
	} else {
		bool useEow{};
		archive.Serialize( useEow );
		if( useEow ) {
			params.EndOfWordToken = LegacyEowToken;
		}
		bool useSow{};
		archive.Serialize( useSow );
		if( useSow ) {
			params.StartOfWordToken = LegacySowToken;
		}
		params.UseRawBytes = false;
		params.UnknownTokenId = DefaultUnknownTokenId;
	}

	tokens.Serialize( archive );
	if( archive.IsLoading() ) {
		ClearCache();
		tokenToId.DeleteAll();
		for( int i = 0; i < tokens.Size(); i++ ) {
			tokenToId.Add( tokens[i], i );
		}
	}
}

void CBytePairEncoder::GetIdToTokenMapping( CMap<int, CString>& output ) const
{
	output.DeleteAll();
	output.SetHashTableSize( Size() );
	output.Add( UnknownTokenId(), UnkToken );
	for( int i = 0; i < tokens.Size(); ++i ) {
		CString tokenUserView = tokens[i];
		output.Add( i + 1 + UnknownTokenId(), tokenUserView );
	}
}

void CBytePairEncoder::GetTokenToIdMapping( CMap<CString, int>& output ) const
{
	output.DeleteAll();
	output.SetHashTableSize( Size() );
	output.Add( UnkToken, UnknownTokenId() );
	for( int i = 0; i < tokens.Size(); ++i ) {
		CString tokenUserView = tokens[i];
		output.Add( tokenUserView, i + 1 + UnknownTokenId() );
	}
}

void CBytePairEncoder::Initialize( const CBPEDictionary& _tokens, const CParams& _params )
{
	NeoAssert( !IsInitialized() );

	// Save parameters and fill maps
	params = _params;
	InitializeUnsafe( _tokens );

	// Check data
	NeoAssert( !UseStartOfWordToken() || params.StartOfWordToken != params.EndOfWordToken );
	bool sowOk = !UseStartOfWordToken() || tokenToId.Has( params.StartOfWordToken );
	bool eowOk = !UseEndOfWordToken() || tokenToId.Has( params.EndOfWordToken );
	// Start-of-Word or End-of-Word tokens must be disabled or must be present in the dictionary
	NeoAssert( sowOk && eowOk );

	const CArray<CString> auxTokens = { params.StartOfWordToken, params.EndOfWordToken };
	for( int i = 0; i < tokens.Size(); ++i ) {
		NeoAssert( isValidToken( tokens[i], auxTokens ) );
	}
}

// Checks that token is a letter, auxiliary or a combination of 2 other tokens
bool CBytePairEncoder::isValidToken( const CString& token, const CArray<CString>& auxTokens ) const
{
	const int charLength = UseRawBytes() ? 1 : GetUtf8CharLength( token[0] );
	if( charLength == token.Length() ) {
		// ok, single letter
		return true;
	}

	if( auxTokens.Has( token ) ) {
		// eow/bow
		return true;
	}

	for( int j = 1; j < token.Length(); ++j ) {
		const CString leftPart = token.Mid( 0, j );
		if( tokenToId.Has( leftPart ) ) {
			const CString rightPart = token.Mid( j, token.Length() - j );
			if( tokenToId.Has( rightPart ) ) {
				return true;
			}
		}
	}
	// The token couldn't be generated with the BPE training procedure.
	return false;
}

void CBytePairEncoder::DoEncode( const CString& word, CArray<int>& tokenIds,
	CArray<int>& tokenLengths ) const
{
	NeoAssert( IsInitialized() );

	CArray<CString> wordTokens;
	CArray<int> wordTokenLengths;
	splitWordIntoInitialTokens( word, wordTokens, &wordTokenLengths );

	while( true ) {
		int bestPairIndex = getShiftedTokenIndex( tokens.Last() ) + 1;
		int bestMergePos = NotFound;
		for( int i = 0; i < wordTokens.Size() - 1; i++ ) {
			const CString pair = mergeTokens( wordTokens[i], wordTokens[i + 1] );
			const int pairIndex = getShiftedTokenIndex( pair );
			if( pairIndex != UnknownTokenId() && pairIndex < bestPairIndex ) {
				bestPairIndex = pairIndex;
				bestMergePos = i;
			}
		}

		if( bestMergePos == NotFound ) {
			break;
		}

		wordTokens[bestMergePos] = mergeTokens( wordTokens[bestMergePos],
			wordTokens[bestMergePos + 1] );
		wordTokenLengths[bestMergePos] += wordTokenLengths[bestMergePos + 1];

		wordTokens.DeleteAt( bestMergePos + 1 );
		wordTokenLengths.DeleteAt( bestMergePos + 1 );
	}

	NeoAssert( wordTokens.Size() == wordTokenLengths.Size() );
	tokenIds.SetBufferSize( tokenIds.Size() + wordTokens.Size() );
	for( int i = 0; i < wordTokens.Size(); i++ ) {
		tokenIds.Add( getShiftedTokenIndex( wordTokens[i] ) );
	}
	tokenLengths.Add( wordTokenLengths );
}

// Returns index of token for encoding.
int CBytePairEncoder::getShiftedTokenIndex( const CString& token ) const
{
	int tokenIndex = NotFound;
	if( tokenToId.Lookup( token, tokenIndex ) ) {
		return tokenIndex + UnknownTokenId() + 1;
	} else {
		// Unknown token
		return UnknownTokenId();
	}
}

// Splits a word into initial tokens: single unicode characters + special tokens (optional).
void CBytePairEncoder::splitWordIntoInitialTokens( const CString& word,
	CArray<CString>& initialTokens, CArray<int>* initialTokensLength ) const
{
	NeoAssert( !word.IsEmpty() );

	if( UseStartOfWordToken() ) {
		initialTokens.Add( params.StartOfWordToken );
	}

	for( int curPos = 0; curPos < word.Length(); ) {
		const int charLength = UseRawBytes() ? 1 : GetUtf8CharLength( word[curPos] );
		NeoAssert( charLength > 0 );
		NeoAssert( curPos + charLength <= word.Length() );
		initialTokens.Add( CString( (const char*)word + curPos, charLength ) );
		curPos += charLength;
	}

	if( UseEndOfWordToken() ) {
		initialTokens.Add( params.EndOfWordToken );
	}

	if( initialTokensLength != nullptr ) {
		NeoAssert( initialTokensLength->IsEmpty() );
		initialTokensLength->Add( 1, initialTokens.Size() );
		if( UseStartOfWordToken() ) {
			initialTokensLength->First() = 0;
		}
		if( UseEndOfWordToken() ) {
			initialTokensLength->Last() = 0;
		}
	}
}

// Concatenates tokens.
CString CBytePairEncoder::mergeTokens( const CString& first, const CString& second )
{
	return first + second;
}
} // namespace NeoML
