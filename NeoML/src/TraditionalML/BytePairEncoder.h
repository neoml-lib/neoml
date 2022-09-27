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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/TraditionalML/SubwordEncoder.h>

namespace NeoML {

// Class that encodes a UTF-8 word using byte-pair-encoding.
class NEOML_API CBytePairEncoder : public IBytePairEncoder {
public:
	// For creating with CreateModel( BytePairEncoderModelName ). Initialize(...) should be used to complete the setup.
	CBytePairEncoder() = default;
	// Construction with an empty dictionary. Generally ctor is used by CBytePairEncoderTrainer.
	CBytePairEncoder( CParams params ) : params( std::move( params ) ) {}
	// Loads a dictionary without additional checks. Completes the initialization for CBytePairEncoderTrainer
	void LoadDictionary( const CBPEDictionary& );
	bool IsInitialized() const { return !tokens.IsEmpty(); }

	// ISubwordEncoder:
	void Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const override;
	int Size() const override { return tokens.Size() + 1; } // One extra for 'Unknown'
	void Serialize( CArchive& archive ) override;

	// IBytePairEncoder:
	void Initialize( const CBPEDictionary& tokens, const CParams& ) override;
	void GetIdToTokenMapping( CMap<int, CString>& ) const override;
	void GetTokenToIdMapping( CMap<CString, int>& ) const override;
	bool UseEndOfWordToken() const override { return !params.EndOfWordToken.IsEmpty(); }
	bool UseStartOfWordToken() const override { return !params.StartOfWordToken.IsEmpty(); }
	bool UseRawBytes() const override { return params.UseRawBytes; }
	int UnknownTokenId() const override { return params.UnknownTokenId; }
	
	// Splits a word into initial tokens: single unicode characters + special tokens (optional).
	void SplitWordIntoInitialTokens( const CString& word, 
		CArray<CString>& initialTokens, CArray<int>* initialTokensLength = nullptr ) const;
	// Concatenates tokens.
	static CString MergeTokens( const CString& first, const CString& second );

protected:
	// ISubwordEncoderWithCache:
	void DoEncode( const CString& word, CArray<int>& tokenIds,
		CArray<int>& tokenLengths ) const override;

private:
	// Index map Id -> Token. Note that the ids are being shifted by UnknownTokenId() + 1 while encoding.
	CBPEDictionary tokens;
	// Reverse Map: Token -> Id. It is an unshifted index (matches 'tokens' array).
	CMap<CString, int> tokenToId;
	// Encoder parameters
	CParams params;

	CString getToken( int shiftedTokenId ) const;
	void removeSpecialTokens( CString& token, bool& hasEow, bool& hasSow ) const;
	bool replaceEowToken( CString& token, const CString& eowToken, const CString& replacement ) const;
	bool replaceSowToken( CString& token, const CString& sowToken, const CString& replacement ) const;
	bool isValidToken( const CString& token, const CArray<CString>& auxTokens ) const;
	int getShiftedTokenIndex( const CString& token ) const;
};

} // namespace NeoML
