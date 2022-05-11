/* Copyright � 2017-2022 ABBYY Production LLC

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
#include <NeoML/TraditionalML/WordDictionary.h>
#include <NeoML/TraditionalML/Model.h>

namespace NeoML {

DECLARE_NEOML_MODEL_NAME( BytePairEncoderModelName, "NeoMLBytePairEncoderModel" )

// Class that encodes a UTF-8 word using byte-pair-encoding.
class NEOML_API CBytePairEncoder : public IBytePairEncoder {
public:
	CBytePairEncoder();
	CBytePairEncoder( const CWordDictionary& tokens, bool useEndOfWordToken, bool useStartOfWordToken );

	// ISubwordEncoder:
	virtual void Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const override;
	virtual int Size() const override;
	virtual void Serialize( CArchive& archive ) override;

	// IBytePairEncoder:
	virtual bool UseEndOfWordToken() const override { return useEndOfWordToken; }
	virtual bool UseStartOfWordToken() const override { return useStartOfWordToken; }

	// Splits a word into initial tokens: single unicode characters + special tokens (optional).
	static void SplitWordIntoInitialTokens( const CString& word, bool useStartOfWordToken,
		bool useEndOfWordToken, CArray<CString>& initialTokens, CArray<int>* initialTokensLength = nullptr );
	// Concatenates tokens.
	static CString MergeTokens( const CString& first, const CString& second );

protected:
	// ISubwordEncoderWithCache:
	virtual void DoEncode( const CString& word, CArray<int>& tokenIds,
		CArray<int>& tokenLengths ) const override;

private:
	// BPE tokens.
	CArray<CString> tokens;
	// Reverse Map: Token -> Token index in tokens array.
	CMap<CString, int> tokenToId;

	// Special flags usage flags.
	bool useEndOfWordToken;
	bool useStartOfWordToken;

	int getTokenIndex( const CString& token ) const;
	CString getToken( int tokenId ) const;

	void removeSpecialTokens( CString& token, bool& hasEoW, bool& hasSoW ) const;
	bool removeEoWToken( CString& token ) const;
	bool removeSoWToken( CString& token ) const;
};

} // namespace NeoML
