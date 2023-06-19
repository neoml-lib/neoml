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

#include <SubwordDecoder.h>

namespace NeoML {

CSubwordDecoder::CSubwordDecoder( ISubwordEncoder::CParams params, CMap<int, CString>&& _idToToken ) :
	params( std::move( params ) )
{
	_idToToken.MoveTo( idToToken );
}

void CSubwordDecoder::Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const
{
	if( tokenIds.IsEmpty() ) {
		return;
	}

	CArray<CString> rawWordTokens;
	rawWordTokens.SetBufferSize( tokenIds.Size() );
	for( int i = 0; i < tokenIds.Size(); i++ ) {
		rawWordTokens.Add( idToToken[tokenIds[i]] );
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

// Removes special subtokens from token.
void CSubwordDecoder::removeSpecialTokens( CString& token, bool& hasEow, bool& hasSow ) const
{
	hasEow = replaceEowToken( token, params.EndOfWordToken, "" );
	hasSow = replaceSowToken( token, params.StartOfWordToken, "" );
}

// If Eow is enabled and if 'token' ends with 'eowToken', replaces matched suffix with 'replacement'
bool CSubwordDecoder::replaceEowToken( CString& token, const CString& eowToken, const CString& replacement ) const
{
	if( params.EndOfWordToken.IsEmpty() || token.Length() < eowToken.Length() ) {
		return false;
	}
	NeoAssert( !eowToken.IsEmpty() );

	const int eowPos = token.Length() - eowToken.Length();
	if( token.CompareSubstr( eowPos, eowToken, eowToken.Length() ) == 0 ) {
		token.StrReplace( eowPos, eowToken.Length(), replacement );
		return true;
	} else {
		return false;
	}
}

// If Sow is enabled and if 'token' starts with 'sowToken', replaces matched prefix with 'replacement'
bool CSubwordDecoder::replaceSowToken( CString& token, const CString& sowToken, const CString& replacement ) const
{
	if( params.StartOfWordToken.IsEmpty() || token.Length() < sowToken.Length() ) {
		return false;
	}
	NeoAssert( !sowToken.IsEmpty() );

	if( token.CompareSubstr( 0, sowToken, sowToken.Length() ) == 0 ) {
		token.StrReplace( 0, sowToken.Length(), replacement );
		return true;
	} else {
		return false;
	}
}
} // namespace NeoML
