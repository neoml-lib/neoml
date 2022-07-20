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
static const CString StartOfWordTokenInternal( "/\xFF" );
static const CString EndOfWordTokenInternal( "\\\xFF" );
static const CString UnkToken( "<UNK>" );

// !!! Do not change:
static constexpr int UnknownTokenId = 0;

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
	const unsigned char byte = ( unsigned char )c;
	return utf8CharacterLength[byte];
}

void CBytePairEncoder::SplitWordIntoInitialTokens( const CString& word, const CString& startOfWordToken,
	const CString& endOfWordToken, CArray<CString>& initialTokens, CArray<int>* initialTokensLength )
{
	NeoAssert( !word.IsEmpty() );

	if( !startOfWordToken.IsEmpty() ) {
		initialTokens.Add( startOfWordToken );
	}

	for( int curPos = 0; curPos < word.Length(); ) {
		const int charLength = getUtf8CharLength( word[curPos] );
		NeoAssert( charLength > 0 );
		NeoAssert( curPos + charLength <= word.Length() );
		initialTokens.Add( CString( ( const char* )word + curPos, charLength ) );
		curPos += charLength;
	}

	if( !endOfWordToken.IsEmpty() ) {
		initialTokens.Add( endOfWordToken );
	}

	if( initialTokensLength != nullptr ) {
		NeoAssert( initialTokensLength->IsEmpty() );
		initialTokensLength->Add( 1, initialTokens.Size() );
		if( !startOfWordToken.IsEmpty() ) {
			initialTokensLength->First() = 0;
		}
		if( !endOfWordToken.IsEmpty() ) {
			initialTokensLength->Last() = 0;
		}
	}
}

CString CBytePairEncoder::MergeTokens( const CString& first, const CString& second )
{
	return first + second;
}

///////////////////////////////////////////////////////////////////////////////

void CBytePairEncoder::Decode( const CArray<int>& tokenIds,
	CArray<CString>& words ) const
{
	if( tokenIds.IsEmpty() ) {
		return;
	}

	CArray<CString> rawWordTokens;
	rawWordTokens.SetBufferSize( tokenIds.Size() );
	for( int i = 0; i < tokenIds.Size(); i++ ) {
		rawWordTokens.Add( getToken( tokenIds[i] ) );
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

int CBytePairEncoder::Size() const
{
	// One extra for 'Unknown'
	return 1 + tokens.Size();
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

void CBytePairEncoder::LoadDictionary( const CWordDictionary& _tokens, 
		const CString& endOfWordToken, const CString& startOfWordToken )
{
	// This particular encoder can be re-initialized
	tokenToId.DeleteAll();
	tokens.DeleteAll();
	ClearCache();

	// Sort words descending
	CWordDictionary finalizedDictionary;
	_tokens.CopyTo( finalizedDictionary );
	finalizedDictionary.Finalize( INT64_MIN );

	// Check data
	NeoAssert( startOfWordToken.IsEmpty() || startOfWordToken != endOfWordToken );
	useStartOfWordToken = !startOfWordToken.IsEmpty();
	useEndOfWordToken = !endOfWordToken.IsEmpty();
	bool sowOk = !useStartOfWordToken || finalizedDictionary.HasWord( startOfWordToken );
	bool eowOk = !useEndOfWordToken || finalizedDictionary.HasWord( endOfWordToken );
	// Start-of-Word or End-of-Word tokens must be disabled or must be present in the dictionary
	NeoAssert( sowOk && eowOk );
	CArray<CString> auxTokens = { startOfWordToken, endOfWordToken };
	// Check that each token is a letter, auxiliary or a combination of 2 other tokens
	const auto& inseparable = findInseparableToken( finalizedDictionary, auxTokens );
	NeoAssert( inseparable.IsEmpty() );

	// Import vocabulary
	for( int i = 0; i < finalizedDictionary.Size(); i++ ) {
		auto token = finalizedDictionary.GetWord( i );
		NeoAssert( !tokenToId.Has( token ) );

		// To encode input texts safely, we replace EOW/SOW with unprintable symbols.
		if( useStartOfWordToken ) {
			const int sowPos = token.Find( startOfWordToken );
			if( sowPos != NotFound ) {
				// Check that Start-of-Word isn't located in the middle of the token
				NeoAssert( token.Find( startOfWordToken, 1 ) == NotFound );
				token.StrReplace( 0, startOfWordToken.Length(), StartOfWordTokenInternal );
			}
		}
		if( useEndOfWordToken ) {
			const int eowPos = token.Find( endOfWordToken );
			if( eowPos != NotFound ) {
				// Check that End-of-Word isn't located anywhere except the end of the token
				NeoAssert( eowPos == token.Length() - endOfWordToken.Length() );
				token.StrReplace( eowPos, endOfWordToken.Length(), EndOfWordTokenInternal );
			}
		}

		tokenToId.Add( token, tokens.Size() );
		tokens.Add( token );
	}
}

// Returns the first inseparable token, if any
CString CBytePairEncoder::findInseparableToken( const CWordDictionary& dictionary, const CArray<CString>& auxTokens )
{
	CArray<bool> isSeparable;
	isSeparable.Add( false, dictionary.Size() );

	for( int i = 0; i < dictionary.Size(); ++i ) {
		const auto& leftToken = dictionary.GetWord( i );

		for( int j = 0; j < dictionary.Size(); ++j ) {
			const auto& rightToken = dictionary.GetWord( j );
			const auto& merge = MergeTokens( leftToken, rightToken );

			const int mergeTokenId = dictionary.GetWordId( merge );
			if( mergeTokenId != NotFound ) {
				isSeparable[mergeTokenId] = true;			
			}
		}
	}

	for( int i = 0; i < dictionary.Size(); ++i ) {
		if( isSeparable[i] ) {
			// ok, token is a combination of two other tokens
			continue;
		}

		auto token = dictionary.GetWord( i );
		const int charLength = getUtf8CharLength( token[0] );
		if( charLength == token.Length() ) {
			// ok, single letter
			continue;
		}

		if( !auxTokens.Has( token ) ) {
			// not a combination, not a single letter, not auxiliary (eow/bow)
			return token;
		}
	}
	return "";
}

void CBytePairEncoder::GetDictionary( CWordDictionary& output, 
	const CString& endOfWordToken, const CString& startOfWordToken ) const
{
	// Check that Start-of-Word and End-of-Word are not used (or, if used, are not empty)
	NeoAssert( !( useStartOfWordToken && startOfWordToken.IsEmpty() ) );
	NeoAssert( !( useEndOfWordToken && endOfWordToken.IsEmpty() ) );

	output.Empty();
	for( int i = 0; i < tokens.Size(); ++i ) {
		CString token = tokens[i];
		if( useStartOfWordToken ) {
			replaceSoWToken( token, StartOfWordTokenInternal, startOfWordToken );
		}
		if( useEndOfWordToken ) {
			replaceEoWToken( token, EndOfWordTokenInternal, endOfWordToken );
		}
		output.AddWord( token, tokens.Size() - i );
	}
}

void CBytePairEncoder::DoEncode( const CString& word, CArray<int>& tokenIds,
	CArray<int>& tokenLengths ) const
{
	CArray<CString> wordTokens;
	CArray<int> wordTokenLengths;
	SplitWordIntoInitialTokens( word, 
		useStartOfWordToken ? StartOfWordTokenInternal : "", 
		useEndOfWordToken ? EndOfWordTokenInternal : "",
		wordTokens, &wordTokenLengths );

	while( true ) {
		int bestPairIndex = tokens.Size() + 1;
		int bestMergePos = NotFound;
		for( int i = 0; i < wordTokens.Size() - 1; i++ ) {
			const CString pair = MergeTokens( wordTokens[i], wordTokens[i + 1] );
			const int pairIndex = getTokenIndex( pair );
			if( pairIndex != UnknownTokenId	&& pairIndex < bestPairIndex ) {
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
	tokenIds.SetBufferSize( tokenIds.Size() + wordTokens.Size() );
	for( int i = 0; i < wordTokens.Size(); i++ ) {
		tokenIds.Add( getTokenIndex( wordTokens[i] ) );
	}
	tokenLengths.Add( wordTokenLengths );
}

// Returns index of token.
int CBytePairEncoder::getTokenIndex( const CString& token ) const
{
	int tokenIndex = NotFound;
	if( tokenToId.Lookup( token, tokenIndex ) ) {
		return tokenIndex + 1;
	} else {
		// Unknown token
		return UnknownTokenId;
	}
}

// Returns string representation of token by tokenId.
CString CBytePairEncoder::getToken( int tokenId ) const
{
	NeoAssert( tokenId >= 0 && tokenId < Size() );

	if( tokenId == UnknownTokenId ) {
		// Unknown token.
		return UnkToken;
	} else {
		return tokens[tokenId - 1];
	}
}

// Removes special subtokens form token.
void CBytePairEncoder::removeSpecialTokens( CString& token, bool& hasEoW, bool& hasSoW ) const
{
	hasEoW = replaceEoWToken( token, EndOfWordTokenInternal, "" );
	hasSoW = replaceSoWToken( token, StartOfWordTokenInternal, "" );
}

bool CBytePairEncoder::replaceEoWToken( CString& token, const CString& eowToken, const CString& replacement ) const
{
	if( !useEndOfWordToken
		|| token.Length() < eowToken.Length() ) 
	{
		return false;
	}

	const int eowPos = token.Length() - eowToken.Length();
	if( token.CompareSubstr( eowPos, eowToken, eowToken.Length() ) == 0 ) {
		token.StrReplace( eowPos, eowToken.Length(), replacement );
		return true;
	} else {
		return false;
	}
}

bool CBytePairEncoder::replaceSoWToken( CString& token, const CString& sowToken, const CString& replacement ) const
{
	if( !useStartOfWordToken
		|| token.Length() < sowToken.Length() ) 
	{
		return false;
	}

	if( token.CompareSubstr( 0, sowToken, sowToken.Length() ) == 0 ) {
		token.StrReplace( 0, sowToken.Length(), replacement );
		return true;
	} else {
		return false;
	}
}

} // namespace NeoML
