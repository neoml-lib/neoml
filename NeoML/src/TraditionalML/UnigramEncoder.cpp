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

#include <UnigramEncoder.h>
#include <SubwordDecoder.h>
#include <NeoML/TraditionalML/GraphGenerator.h>

namespace NeoML {

REGISTER_NEOML_MODEL( CUnigramEncoder, UnigramEncoderModelName )

const CString CUnigramEncoder::unkTokenName( "<UNK>" );

CArchive& operator>>( CArchive& archive, IUnigramEncoder::CSubword& token )
{
	archive.Serialize( token.Text );
	archive.Serialize( token.Score );
	return archive;
}

CArchive& operator<<( CArchive& archive, IUnigramEncoder::CSubword& token )
{
	archive.Serialize( token.Text );
	archive.Serialize( token.Score );
	return archive;
}

// ----------------

void CUnigramEncoder::Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const
{
	NeoAssert( IsInitialized() );
	if( decoder == nullptr ) {
		CMap<int, CString> idToTokenOut;
		GetIdToTokenMapping( idToTokenOut );
		decoder = std::make_unique<CSubwordDecoder>( params, std::move( idToTokenOut ) );
	}
	decoder->Decode( tokenIds, words );
}

void CUnigramEncoder::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );

	params.Serialize( archive );
	idToToken.Serialize( archive );
	NeoAssert( !idToToken.IsEmpty() );

	if( archive.IsLoading() ) {
		ClearCache();
		tokenTrie.DeleteAll();
		tokenToId.DeleteAll();

		tokenTrie.Set( idToToken[0] );
		for( int i = 1; i < idToToken.Size(); ++i ) {
			auto* token = idToToken[i];
			tokenToId.Add( token->Text, i + UnknownTokenId() );
			tokenTrie.Add( token->Text )->Set( token );
		}
	}
}

void CUnigramEncoder::GetIdToTokenMapping( CMap<int, CString>& output ) const
{
	output.DeleteAll();
	output.SetHashTableSize( Size() );
	for( int i = 0; i < idToToken.Size(); ++i ) {
		output.Add( i + UnknownTokenId(), idToToken[i]->Text );
	}
}

void CUnigramEncoder::GetTokenToIdMapping( CMap<CString, int>& output ) const
{
	tokenToId.CopyTo( output );
	output.Add( unkTokenName, UnknownTokenId() );
}

void CUnigramEncoder::Initialize( const CUnigramDictionary& tokens, const CParams& _params )
{
	NeoAssert( !IsInitialized() );
	params = _params;

	idToToken.SetBufferSize( tokens.Size() + 1 );
	for( const auto& token : tokens ) {
		idToToken.Add( new CSubword( token ) );
	}
	idToToken.QuickSort<DescendingPtrByMember<CSubword, double, &CSubword::Score>>();
	idToToken.InsertAt( new CSubword( unkTokenName, unkTokenScore ), 0 );
	tokenTrie.Set( idToToken[0] );

	for( int i = 1; i < idToToken.Size(); ++i ) {
		const auto& token = *idToToken[i];
		NeoAssert( !tokenToId.Has( token.Text ) );
		tokenToId.Add( token.Text, i + UnknownTokenId() );
		tokenTrie.Add( token.Text )->Set( idToToken.Last() );
	}
}

void CUnigramEncoder::GetDictionary( CUnigramDictionary& output ) const
{
	output.DeleteAll();
	output.SetBufferSize( idToToken.Size() - 1 );
	for( int i = 1; i < idToToken.Size(); ++i ) {
		output.Add( { idToToken[i]->Text, idToToken[i]->Score } );
	}
}

void CUnigramEncoder::DoEncode( const CString& word, CArray<int>& tokenIds, CArray<int>& tokenLengths ) const
{
	NeoAssert( IsInitialized() );

	const int firstTokenPos = tokenLengths.Size();

	const CString inputWithBorders = params.StartOfWordToken + word + params.EndOfWordToken;
	if( inputWithBorders.IsEmpty() ) {
		return;
	}
	CPointerArray<CSubwordLdGraphArc> tokenSegments;
	CSubwordLdGraph tokenStructure( inputWithBorders );
	FillSubwordLdGraphFromTrie( inputWithBorders, &tokenTrie, tokenSegments, tokenStructure );

	CGraphGenerator<CSubwordLdGraph> graphGen( &tokenStructure, 0.0, -FLT_MAX / 2 );

	NeoAssert( graphGen.CanGenerateNextPath() );
	CArray<const CSubwordLdGraphArc*> path;
	graphGen.GetNextPath( path );
	for( const CSubwordLdGraphArc* segment : path ) {
		const CString& tokenText = segment->Arc->Text;
		tokenIds.Add( getTokenIndex( tokenText ) );
		tokenLengths.Add( tokenText.Length() );
	}
	tokenLengths[firstTokenPos] -= params.StartOfWordToken.Length();
	tokenLengths.Last() -= params.EndOfWordToken.Length();
}

// Returns index of token for encoding.
int CUnigramEncoder::getTokenIndex( const CString& token ) const
{
	int tokenIndex = UnknownTokenId();
	tokenToId.Lookup( token, tokenIndex );
	return tokenIndex;
}
} // namespace NeoML
