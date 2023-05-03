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
#include <NeoML/TraditionalML/SubwordEncoder.h>
#include <SubwordDecoder.h>
#include <memory>

namespace NeoML {
class CSubwordDecoder;

// Trie based on char (CString) letters. Can store any class in terminal nodes
template <class T>
class NEOML_API CTrieNode {
public:
	CTrieNode() = default;
	CTrieNode( const CTrieNode& other ) = delete;
	~CTrieNode() { DeleteAll(); }

	const CTrieNode* Go( char letter ) const;
	const T* Get() const { return data; }
	const T* Get( const CString& text ) const;
	void Add( const CString& text, T* data );
	void DeleteAll();

private:
	// Transitions
	CMap<char, CTrieNode*> children;
	// Any data that correspond to this node as a terminal (no ownership)
	T* data = nullptr;
};

// Implementation of the subword encoding using the Unigram algorithm
class NEOML_API CUnigramEncoder : public IUnigramEncoder {
public:
	// ISubwordEncoder:
	void Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const override;
	int Size() const override { return tokenStorage.Size(); }
	void Serialize( CArchive& ) override;
	void GetIdToTokenMapping( CMap<int, CString>& ) const override;
	void GetTokenToIdMapping( CMap<CString, int>& ) const override;
	bool UseEndOfWordToken() const override { return !params.EndOfWordToken.IsEmpty(); }
	bool UseStartOfWordToken() const override { return !params.StartOfWordToken.IsEmpty(); }
	bool UseRawBytes() const override { return params.UseRawBytes; }
	int UnknownTokenId() const override { return params.UnknownTokenId; }

	// IUnigramEncoder:
	void Initialize( const CUnigramDictionary& tokens, const CParams& ) override;
	bool IsInitialized() const override { return !tokenStorage.IsEmpty(); }
	void GetDictionary( CUnigramDictionary& tokens ) const override;

protected:
	~CUnigramEncoder() override = default;
	// ISubwordEncoderWithCache:
	void DoEncode( const CString& word, CArray<int>& tokenIds, CArray<int>& tokenLengths ) const override;

private:
	static const CString unkTokenName;
	static constexpr double unkTokenScore = -10.0;

	// Encoder parameters
	CParams params;
	// Structures filled with the vocabulary tokens
	CMap<CString, int> tokenToId;
	CTrieNode<const CSubtoken> tokenTrie;
	CPointerArray<CSubtoken> tokenStorage;
	// Lazy-initialized mechanism for Decode() function
	mutable std::unique_ptr<CSubwordDecoder> decoder = nullptr;

	int getShiftedTokenIndex( const CString& token ) const;
};

// ------------------------------------

template <class T>
const CTrieNode<T>* CTrieNode<T>::Go( char letter ) const
{
	CTrieNode* res = nullptr;
	children.Lookup( letter, res );
	return res;
}

template <class T>
const T* CTrieNode<T>::Get( const CString& text ) const
{
	const auto* pos = this;
	for( int i = 0; i < text.Length(); ++i ) {
		pos = pos->Go( text[i] );
		if( pos == nullptr ) {
			return nullptr;
		}
	}
	return pos->data;
}

template <class T>
void CTrieNode<T>::Add( const CString& text, T* _data )
{
	auto* pos = this;
	for( int i = 0; i < text.Length(); ++i ) {
		char c = text[i];
		CTrieNode* next = nullptr;
		if( !pos->children.Lookup( c, next ) ) {
			next = new CTrieNode;
			pos->children.Add( c, next );
		}
		pos = next;
	}
	pos->data = _data;
}

template <class T>
void CTrieNode<T>::DeleteAll()
{
	for( auto pos = children.GetFirstPosition(); pos != NotFound; pos = children.GetNextPosition( pos ) ) {
		delete children.GetValue( pos );
	}
	data = nullptr;
}
} // namespace NeoML
