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

#include <BytePairEncoder.h>

namespace NeoML {

REGISTER_NEOML_MODEL( CBytePairEncoder, BytePairEncoderModelName )

// Some special tokens.
static const CString StartOfWordToken( "/\xFF" );
static const CString EndOfWordToken( "\\\xFF" );

static const CString UnknownToken( "<UNK>" );

// Based on Utf8FirstByteProperties from UtfConverterFO.h.
static constexpr int utf8CharacterLength[256] = {
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 00-0F
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 10-1F
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 20-2F
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 30-3F
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 40-4F
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 50-5F
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 60-6F
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 70-7F
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 80-8F
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 90-9F
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // A0-AF
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // B0-BF
	0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // C0-CF
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // D0-DF
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // E0-EF
	4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // F0-FF
};

// Returns the length of character utf8 encoding by the first byte.
static int getUtf8CharLength( char c )
{
	const unsigned char byte = ( unsigned char ) c;
	return utf8CharacterLength[byte];
}

void CBytePairEncoder::SplitWordIntoInitialTokens( const CString& word, bool useStartOfWordToken,
	bool useEndOfWordToken, CArray<CString>& initialTokens, CArray<int>* initialTokensLength )
{
	NeoAssert( !word.IsEmpty() );

	if( useStartOfWordToken ) {
		initialTokens.Add( StartOfWordToken );
	}

	for( int curPos = 0; curPos < word.Length(); ) {
		const int charLength = getUtf8CharLength( word[static_cast< unsigned int >( curPos )] );
		NeoAssert( charLength > 0 );
		NeoAssert( curPos + charLength <= word.Length() );
		initialTokens.Add( CString( ( const char* )word + curPos, charLength ) );
		curPos += charLength;
	}

	if( useEndOfWordToken ) {
		initialTokens.Add( EndOfWordToken );
	}

	if( initialTokensLength != nullptr ) {
		NeoAssert( initialTokensLength->IsEmpty() );
		initialTokensLength->Add( 1, initialTokens.Size() );
		if( useStartOfWordToken ) {
			initialTokensLength->First() = 0;
		}
		if( useEndOfWordToken ) {
			initialTokensLength->Last() = 0;
		}
	}
}

CString CBytePairEncoder::MergeTokens( const CString& first, const CString& second )
{
	return first + second;
}

///////////////////////////////////////////////////////////////////////////////

CBytePairEncoder::CBytePairEncoder() :
	useEndOfWordToken( false ),
	useStartOfWordToken( false )
{}

CBytePairEncoder::CBytePairEncoder( const CWordDictionary& tokens_, bool useEndOfWordToken, 
		bool useStartOfWordToken ) :
	useEndOfWordToken( useEndOfWordToken ),
	useStartOfWordToken( useStartOfWordToken )
{
	for( int i = 0; i < tokens_.Size(); i++ ) {
		const CString newToken = tokens_.GetWord( i );
		NeoAssert( !tokenToId.Has( newToken ) );
		tokenToId.Add( newToken, tokens.Size() );
		tokens.Add( newToken );
	}
}

void CBytePairEncoder::Decode( const CArray<int>& tokenIds,
	CArray<CString>& words ) const
{
	if( tokenIds.IsEmpty() ) {
		return;
	}

	CArray<CString> rawWordTokens;
	for( int i = 0; i < tokenIds.Size(); i++ ) {
		if( tokenIds[i] == NotFound ) {
			rawWordTokens.Add( UnknownToken );
		} else {
			rawWordTokens.Add( tokens[tokenIds[i]] );
		}
	}

	CArray<bool> isWordBorder;
	isWordBorder.Add( false, rawWordTokens.Size() - 1 );

	for( int i = 0; i < rawWordTokens.Size(); i++ ) {
		bool hasEow = false;
		bool hasSow = false;
		removeSpecialTokens( rawWordTokens[i], hasEow, hasSow );
		if( i > 0 ) {
			isWordBorder[i - 1] |= hasSow;
		}
		if( i < rawWordTokens.Size() - 1 ) {
			isWordBorder[i] |= hasEow;
		}
	}

	CString currentWord;
	for( int i = 0; i < rawWordTokens.Size(); i++ ) {
		currentWord += rawWordTokens[i];
		if( i < rawWordTokens.Size() - 1
			&& isWordBorder[i] ) 
		{
			words.Add( currentWord );
			currentWord = "";
		}
	}
	words.Add( currentWord );
}

void CBytePairEncoder::GetTokenIdRange( int& minId, int& maxId ) const
{
	minId = NotFound;
	maxId = tokens.Size() - 1;
}

void CBytePairEncoder::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );
	archive.Serialize( useStartOfWordToken );
	archive.Serialize( useEndOfWordToken );
	tokens.Serialize( archive );
	if( archive.IsLoading() ) {
		tokenToId.Empty();
		for( int i = 0; i < tokens.Size(); i++ ) {
			tokenToId.Add( tokens[i], i );
		}
	}
}

void CBytePairEncoder::doEncode( const CString& word, CArray<int>& tokenIds,
	CArray<int>& tokenLengths ) const
{
	CArray<CString> wordTokens;
	CArray<int> wordTokenLengths;
	SplitWordIntoInitialTokens( word, useStartOfWordToken, useEndOfWordToken,
		wordTokens, &wordTokenLengths );

	while( true ) {
		int bestPairIndex = tokens.Size();
		int bestMergePos = NotFound;
		for( int i = 0; i < wordTokens.Size() - 1; i++ ) {
			const CString pair = MergeTokens( wordTokens[i], wordTokens[i + 1] );
			const int pairIndex = getTokenIndex( pair );
			if( pairIndex != NotFound
				&& pairIndex < bestPairIndex )
			{
				bestPairIndex = pairIndex;
				bestMergePos = i;
			}
		}

		if( bestMergePos == NotFound ) {
			break;
		}

		wordTokens[bestMergePos] = MergeTokens( wordTokens[bestMergePos],
			wordTokens[bestMergePos + 1] );
		wordTokenLengths[bestMergePos] += wordTokenLengths[bestMergePos + 1];

		wordTokens.DeleteAt( bestMergePos + 1 );
		wordTokenLengths.DeleteAt( bestMergePos + 1 );
	}

	NeoAssert( wordTokens.Size() == wordTokenLengths.Size() );
	for( int i = 0; i < wordTokens.Size(); i++ ) {
		tokenIds.Add( getTokenIndex( wordTokens[i] ) );
	}
	tokenLengths.Add( wordTokenLengths );
}

// Returns index of token (-1 if not found).
int CBytePairEncoder::getTokenIndex( const CString& token ) const
{
	int tokenIndex = NotFound;
	tokenToId.Lookup( token, tokenIndex );
	return tokenIndex;
}

// Removes special subtokens form token.
void CBytePairEncoder::removeSpecialTokens( CString& token, bool& hasEoW, bool& hasSoW ) const
{
	hasEoW = removeEoWToken( token );
	hasSoW = removeSoWToken( token );
}

bool CBytePairEncoder::removeEoWToken( CString& token ) const
{
	if( !useEndOfWordToken
		|| token.Length() < EndOfWordToken.Length() ) 
	{
		return false;
	}

	const int cleanLength = token.Length() - EndOfWordToken.Length();
	const CString suffix( ( const char* )token + cleanLength, EndOfWordToken.Length() );
	if( suffix == EndOfWordToken ) {
		token = CString( ( const char* )token, cleanLength );
		return true;
	} else {
		return false;
	}
}

bool CBytePairEncoder::removeSoWToken( CString& token ) const
{
	if( !useStartOfWordToken
		|| token.Length() < StartOfWordToken.Length() ) 
	{
		return false;
	}

	const int cleanLength = token.Length() - StartOfWordToken.Length();
	const CString prefix( ( const char* )token, StartOfWordToken.Length() );
	if( prefix == StartOfWordToken ) {
		token = CString( ( const char* )token + StartOfWordToken.Length(), cleanLength );
		return true;
	} else {
		return false;
	}
}

} // namespace NeoML
