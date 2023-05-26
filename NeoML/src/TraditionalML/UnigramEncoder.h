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

#include <NeoML/TraditionalML/SubwordEncoder.h>
#include <SubwordDecoder.h>
#include <UnigramTools.h>
#include <memory>

namespace NeoML {
class CSubwordDecoder;

// Implementation of the subword encoding using the Unigram algorithm
class NEOML_API CUnigramEncoder : public IUnigramEncoder {
public:
	// ISubwordEncoder:
	void Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const override;
	int Size() const override { return idToToken.Size(); }
	void Serialize( CArchive& ) override;
	void GetIdToTokenMapping( CMap<int, CString>& ) const override;
	void GetTokenToIdMapping( CMap<CString, int>& ) const override;
	bool UseEndOfWordToken() const override { return !params.EndOfWordToken.IsEmpty(); }
	bool UseStartOfWordToken() const override { return !params.StartOfWordToken.IsEmpty(); }
	bool UseRawBytes() const override { return params.UseRawBytes; }
	int UnknownTokenId() const override { return params.UnknownTokenId; }

	// IUnigramEncoder:
	void Initialize( const CUnigramDictionary& tokens, const CParams& ) override;
	bool IsInitialized() const override { return !idToToken.IsEmpty(); }
	void GetDictionary( CUnigramDictionary& tokens ) const override;

protected:
	~CUnigramEncoder() override = default;
	// ISubwordEncoderWithCache:
	void DoEncode( const CString& word, CArray<int>& tokenIds, CArray<int>& tokenLengths ) const override;

private:
	static const CString unkTokenName;
	static constexpr double unkTokenScore = -20.0;

	// Encoder parameters
	CParams params;
	// Structures filled with the vocabulary tokens
	CMap<CString, int> tokenToId;
	CPointerArray<CSubword> idToToken;
	CTrieNode<CSubword*> tokenTrie;
	// Lazy-initialized mechanism for Decode() function
	mutable std::unique_ptr<CSubwordDecoder> decoder = nullptr;

	int getTokenIndex( const CString& token ) const;
};
} // namespace NeoML
