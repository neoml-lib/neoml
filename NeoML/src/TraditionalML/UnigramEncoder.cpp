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
#include <NeoML/TraditionalML/LdGraph.h>
#include <cfloat>

namespace NeoML {

REGISTER_NEOML_MODEL( CUnigramEncoder, UnigramEncoderModelName )

const CString CUnigramEncoder::unkTokenName( "<UNK>" );

CArchive& operator>>( CArchive& archive, IUnigramEncoder::CSubtoken& token )
{
	archive.Serialize( token.Text );
	archive.Serialize( token.Score );
	return archive;
}

CArchive& operator<<( CArchive& archive, IUnigramEncoder::CSubtoken& token )
{
	archive.Serialize( token.Text );
	archive.Serialize( token.Score );
	return archive;
}

// ----------------

// Edge in CTokenLdGraph i.e. token from the vocabulary
struct CTokenLdGraphArc {
	using Quality = double;

	CTokenLdGraphArc() = default;
	CTokenLdGraphArc( int begin, int end, const IUnigramEncoder::CSubtoken* arc ) :
		Begin( begin ), End( end ), Arc( arc )
	{
		if( arc != nullptr ) {
			Cost = arc->Score;
		}
	}

	int Begin = -1;
	int End = 1;
	double Cost = -10;
	const IUnigramEncoder::CSubtoken* Arc = nullptr;

	// CLdGraph
	int InitialCoord() const { return Begin; }
	int FinalCoord() const { return End; }
	Quality ArcQuality() const { return Cost; }
};

// Graph represents all possible tokenizations (splits) of some word
class CTokenLdGraph : public CLdGraph<CTokenLdGraphArc> {
public:
	using GraphArc = CTokenLdGraphArc;

	CTokenLdGraph() = default;
	explicit CTokenLdGraph( const CString& _text ) :
		CLdGraph( 0, max( 1, _text.Length() ) ) {}
	~CTokenLdGraph() override { DetachAll(); }
};

// create LdGraph with all possible splits of 'input' word
static void fillTokenLdGraphFromTrie( const CString& input,
	const CTrieNode<const IUnigramEncoder::CSubtoken>* trie,
	CPointerArray<CTokenLdGraphArc>& tokenSegments,
	CTokenLdGraph& tokenStructure )
{
	CArray<bool> isSymbolCovered;
	isSymbolCovered.Add( false, input.Length() );

	// traverse trie with 'input' to find all terminals (subtokens)
	for( int begin = 0; begin < input.Length(); ++begin ) {
		const auto* triePos = trie;
		for( int i = begin; i < input.Length(); ++i ) {
			triePos = triePos->Go( input[i] );
			if( triePos == nullptr ) {
				break;
			}
			const auto* token = triePos->Get();
			if( token != nullptr ) {
				tokenSegments.Add( new CTokenLdGraphArc( begin, i + 1, token ) );
				tokenStructure.InsertArc( tokenSegments.Last() );
				for( int pos = begin; pos <= i; ++pos ) {
					isSymbolCovered[pos] = true;
				}
			}
		}
	}

	// cover unknown letters with <UNK> symbol
	const auto* unkToken = trie->Get();
	NeoAssert( unkToken != nullptr );
	for( int i = 0; i < input.Length(); ++i ) {
		if( !isSymbolCovered[i] ) {
			tokenSegments.Add( new CTokenLdGraphArc( i, i + 1, unkToken ) );
			tokenStructure.InsertArc( tokenSegments.Last() );
		}
	}

	tokenStructure.CalculateBestPathQuality( -FLT_MAX / 2 );
}

// ----------------

void CUnigramEncoder::Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const
{
	NeoAssert( IsInitialized() );
	if( decoder == nullptr ) {
		CMap<int, CString> idToToken;
		GetIdToTokenMapping( idToToken );
		decoder = std::make_unique<CSubwordDecoder>( params, std::move( idToToken ) );
	}
	decoder->Decode( tokenIds, words );
}

void CUnigramEncoder::Serialize( CArchive& archive )
{
	ClearCache();

	params.Serialize( archive );
	tokenStorage.Serialize( archive );

	if( archive.IsLoading() ) {
		tokenTrie.DeleteAll();
		tokenToId.DeleteAll();

		tokenTrie.Add( "", tokenStorage[0] );
		for( int i = 1; i < tokenStorage.Size(); ++i ) {
			const auto* token = tokenStorage[i];
			tokenToId.Add( token->Text, tokenStorage.Size() );
			tokenTrie.Add( token->Text, token );
		}
	}
}

void CUnigramEncoder::GetIdToTokenMapping( CMap<int, CString>& output ) const
{
	output.DeleteAll();
	output.SetHashTableSize( Size() );
	for( int i = 0; i < tokenStorage.Size(); ++i ) {
		output.Add( i + UnknownTokenId(), tokenStorage[i]->Text );
	}
}

void CUnigramEncoder::GetTokenToIdMapping( CMap<CString, int>& output ) const
{
	output.DeleteAll();
	output.SetHashTableSize( Size() );
	for( int i = 0; i < tokenStorage.Size(); ++i ) {
		output.Add( tokenStorage[i]->Text, i + UnknownTokenId() );
	}
}

void CUnigramEncoder::Initialize( const CUnigramDictionary& tokens, const CParams& _params )
{
	NeoAssert( !IsInitialized() );
	params = _params;

	tokenStorage.Add( new CSubtoken( unkTokenName, unkTokenScore ) );
	tokenTrie.Add( "", tokenStorage[0] );

	for( const auto& token : tokens ) {
		tokenToId.Add( token.Text, tokenStorage.Size() );
		tokenStorage.Add( new CSubtoken( token ) );
		tokenTrie.Add( token.Text, tokenStorage.Last() );
	}
}

void CUnigramEncoder::GetDictionary( CUnigramDictionary& output ) const
{
	output.DeleteAll();
	output.SetBufferSize( tokenStorage.Size() - 1 );
	for( int i = 1; i < tokenStorage.Size(); ++i ) {
		output.Add( { tokenStorage[i]->Text, tokenStorage[i]->Score } );
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
	CPointerArray<CTokenLdGraphArc> tokenSegments;
	CTokenLdGraph tokenStructure( inputWithBorders );
	fillTokenLdGraphFromTrie( inputWithBorders, &tokenTrie, tokenSegments, tokenStructure );

	CGraphGenerator<CTokenLdGraph> graphGen( &tokenStructure, 0.0, -FLT_MAX / 2 );

	NeoAssert( graphGen.CanGenerateNextPath() );
	CArray<const CTokenLdGraphArc*> path;
	graphGen.GetNextPath( path );
	for( const CTokenLdGraphArc* segment : path ) {
		const CString& tokenText = segment->Arc->Text;
		tokenIds.Add( getShiftedTokenIndex( tokenText ) );
		tokenLengths.Add( tokenText.Length() );
	}
	tokenLengths[firstTokenPos] -= params.StartOfWordToken.Length();
	tokenLengths.Last() -= params.EndOfWordToken.Length();
}

// Returns index of token for encoding.
int CUnigramEncoder::getShiftedTokenIndex( const CString& token ) const
{
	int tokenIndex = NotFound;
	if( tokenToId.Lookup( token, tokenIndex ) ) {
		return tokenIndex + UnknownTokenId();
	} else {
		// Unknown token
		return UnknownTokenId();
	}
}
} // namespace NeoML
