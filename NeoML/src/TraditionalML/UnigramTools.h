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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/TraditionalML/LdGraph.h>
#include <NeoML/TraditionalML/SubwordEncoder.h>
#include <cfloat>

namespace NeoML {

// Trie based on char (CString) letters. Can store any data in nodes
template <class T>
class NEOML_API CTrieNode {
public:
	CTrieNode() = default;
	CTrieNode( const CTrieNode& other ) = delete;
	~CTrieNode() { DeleteAll(); }

	CTrieNode* Go( char letter );
	const CTrieNode* Go( char letter ) const { return const_cast<CTrieNode*>(this)->Go( letter ); }

	CTrieNode* Go( const CString& text );
	const CTrieNode* Go( const CString& text ) const { return const_cast<CTrieNode*>(this)->Go( text ); }

	CTrieNode* Add( char letter );
	CTrieNode* Add( const CString& text );

	const T& Get() const { return data; }
	T& Get() { return data; }
	void Set( T _data ) { data = std::move( _data ); }

	void DeleteAll();

	// CMap-like iteration functions over children (letter transitions)
	TMapPosition GetFirstChildPos() const { return children.GetFirstPosition(); }
	TMapPosition GetNextChildPos( TMapPosition p ) const { return children.GetNextPosition( p ); }
	CTrieNode* GetChild( TMapPosition p ) { return children.GetValue( p ); }
	const CTrieNode* GetChild( TMapPosition p ) const { return children.GetValue( p ); }

private:
	// Transitions (owned)
	CMap<char, CTrieNode*> children;
	// Any data that corresponds to this node (no ownership)
	// Consider replacement with COptional<T> one day...
	T data{};
};

// Edge in CSubwordLdGraphArc i.e. token from the vocabulary
struct CSubwordLdGraphArc {
	using Quality = double;

	CSubwordLdGraphArc() = default;
	CSubwordLdGraphArc( int begin, int end, const IUnigramEncoder::CSubword* arc );

	int Begin = -1;
	int End = 1;
	double Cost = -10;
	const IUnigramEncoder::CSubword* Arc = nullptr;

	// CLdGraph
	int InitialCoord() const { return Begin; }
	int FinalCoord() const { return End; }
	Quality ArcQuality() const { return Cost; }
};

// Graph representing all possible tokenizations (splits) of some word
class CSubwordLdGraph : public CLdGraph<CSubwordLdGraphArc> {
public:
	using GraphArc = CSubwordLdGraphArc;

	CSubwordLdGraph() = default;
	explicit CSubwordLdGraph( const CString& word ) :
		CLdGraph( 0, max( 1, word.Length() ) ) {}
	~CSubwordLdGraph() override { DetachAll(); }
};

//------------------

// Constructs a graph of all possible tokenizations of the word based on tokens from the trie
template <class Trie>
void FillSubwordLdGraphFromTrie( const CString& word,
	const Trie* trie, // CTrieNode<CUnigramTrainer::CTrainingSubword> or CTrieNode<IUnigramEncoder::CSubword>
	CPointerArray<CSubwordLdGraphArc>& subwordSegments,
	CSubwordLdGraph& subwordStructure )
{
	CArray<bool> isSymbolCovered;
	isSymbolCovered.Add( false, word.Length() );

	// traverse trie with 'input' to find all terminals (subtokens)
	for( int begin = 0; begin < word.Length(); ++begin ) {
		const auto* node = trie;
		for( int i = begin; i < word.Length(); ++i ) {
			node = node->Go( word[i] );
			if( node == nullptr ) {
				break;
			}
			const auto* token = node->Get();
			if( token != nullptr ) {
				subwordSegments.Add( new CSubwordLdGraphArc( begin, i + 1, token ) );
				subwordStructure.InsertArc( subwordSegments.Last() );
				for( int pos = begin; pos <= i; ++pos ) {
					isSymbolCovered[pos] = true;
				}
			}
		}
	}

	// Cover unknown letters with <UNK> symbol.
	// It can be nullptr during training, but all letters are known there
	// NeoAssert( unkToken != nullptr ) is inside new CSubwordLdGraphArc
	const auto* unkToken = trie->Get();
	for( int i = 0; i < word.Length(); ++i ) {
		if( !isSymbolCovered[i] ) {
			subwordSegments.Add( new CSubwordLdGraphArc( i, i + 1, unkToken ) );
			subwordStructure.InsertArc( subwordSegments.Last() );
		}
	}

	subwordStructure.CalculateBestPathQuality( -FLT_MAX / 2 );
}

//------------------

template <class T>
CTrieNode<T>* CTrieNode<T>::Go( char letter )
{
	CTrieNode* res = nullptr;
	children.Lookup( letter, res );
	return res;
}

template <class T>
CTrieNode<T>* CTrieNode<T>::Go( const CString& text )
{
	auto* pos = this;
	for( int i = 0; i < text.Length(); ++i ) {
		pos = pos->Go( text[i] );
		if( pos == nullptr ) {
			return nullptr;
		}
	}
	return pos;
}

template <class T>
CTrieNode<T>* CTrieNode<T>::Add( char letter )
{
	auto* next = Go( letter );
	if( next == nullptr ) {
		next = new CTrieNode;
		children.Add( letter, next );
	}
	return next;
}

template <class T>
CTrieNode<T>* CTrieNode<T>::Add( const CString& text )
{
	auto* pos = this;
	for( int i = 0; i < text.Length(); ++i ) {
		pos = pos->Add( text[i] );
	}
	return pos;
}

template <class T>
void CTrieNode<T>::DeleteAll()
{
	for( auto p = children.GetFirstPosition(); p != NotFound; p = children.GetNextPosition( p ) ) {
		delete children.GetValue( p );
	}
	children.DeleteAll();
	data = T();
}
} // namespace NeoML
