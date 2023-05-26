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
#include <SubwordDecoder.h>
#include <memory>

namespace NeoML {

class CSubwordDecoder;

// Class that encodes a UTF-8 word using byte-pair-encoding.
class NEOML_API CBytePairEncoder : public IBytePairEncoder {
public:
	// For creating with CreateModel( BytePairEncoderModelName ). Initialize(...) should be used to complete the setup.
	CBytePairEncoder() = default;
	// Construction with an empty dictionary. Generally ctor is used by CBytePairEncoderTrainer.
	CBytePairEncoder( CParams params ) : params( std::move( params ) ) {}
	// Loads a dictionary without additional checks. Completes the initialization for CBytePairEncoderTrainer
	void InitializeUnsafe( const CBPEDictionary& );

	// ISubwordEncoder:
	void Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const override;
	int Size() const override { return tokens.Size() + 1; } // One extra for 'Unknown'
	void Serialize( CArchive& archive ) override;
	void GetIdToTokenMapping( CMap<int, CString>& ) const override;
	void GetTokenToIdMapping( CMap<CString, int>& ) const override;
	bool UseEndOfWordToken() const override { return !params.EndOfWordToken.IsEmpty(); }
	bool UseStartOfWordToken() const override { return !params.StartOfWordToken.IsEmpty(); }
	bool UseRawBytes() const override { return params.UseRawBytes; }
	int UnknownTokenId() const override { return params.UnknownTokenId; }

	// IBytePairEncoder:
	void Initialize( const CBPEDictionary& tokens, const CParams& ) override;
	bool IsInitialized() const override { return !tokens.IsEmpty(); }

protected:
	~CBytePairEncoder() override = default;
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
	// Lazy-initialized mechanism for Decode() function
	mutable std::unique_ptr<CSubwordDecoder> decoder;

	bool isValidToken( const CString& token, const CArray<CString>& auxTokens ) const;
	int getShiftedTokenIndex( const CString& token ) const;
	void splitWordIntoInitialTokens( const CString& word, 
		CArray<CString>& initialTokens, CArray<int>* initialTokensLength = nullptr ) const;
	static CString mergeTokens( const CString& first, const CString& second );
};

} // namespace NeoML
