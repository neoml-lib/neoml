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
#include <NeoML/TraditionalML/Model.h>

namespace NeoML {

// An encoder tokenizes input sequence with parts of words ('subwords') as tokens.
class NEOML_API ISubwordEncoder : virtual public IObject {
public:
	static constexpr int DefaultUnknownTokenId = 0;

	struct CParams {
		// End-of-Word (EOW), a string that will be added to the end of each input word.
		CString EndOfWordToken;
		// Start-of-Word (SOW), a string that will be added to the beginning of each input word.
		CString StartOfWordToken;
		// Treat strings as arrays of raw bytes,
		// which decreases the maximum size of the initial vocabulary to 256 and allows to completely avoid unknown symbols.
		bool UseRawBytes = false;
		// The id of <UNK>.
		// All other tokens are continuously enumerated from 'UnknownTokenId' + 1. Ids [0, UnknownTokenId) are not used when encoding.
		int UnknownTokenId = DefaultUnknownTokenId;

		CParams() = default;
		CParams( CString endOfWordToken, CString startOfWordToken, bool useRawBytes, int unknownTokenId );

		void Serialize( CArchive& archive );
	};

	// Encodes a word as a sequence of token ids with corresponding token lengths.
	// TokenId range = [0, ... , Size() - 1].
	// To encode a string with wide characters you have to first encode it as utf-8 and wrap it in CString.
	// In this case 'tokenLengths' will contain lengths of the tokens according to the original string version.
	virtual void Encode( const CString& word, CArray<int>& tokenIds,
		CArray<int>& tokenLengths ) const = 0;

	// Decodes sequence of token ids into a sequence of words.
	virtual void Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const = 0;

	// Returns number of tokens.
	virtual int Size() const = 0;

	// Serializes the model
	void Serialize( CArchive& ) override = 0;

	// The functions below should not be used before initialization.
	// Returns mappings as they are performed by the encoder
	virtual void GetIdToTokenMapping( CMap<int, CString>& ) const = 0;
	virtual void GetTokenToIdMapping( CMap<CString, int>& ) const = 0;

	// Encoder parameters getters:
	virtual bool UseEndOfWordToken() const = 0;
	virtual bool UseStartOfWordToken() const = 0;
	virtual bool UseRawBytes() const = 0;
	virtual int UnknownTokenId() const = 0;
};

// Subword encoder which supports caching results of 'Encode' calls.
class NEOML_API ISubwordEncoderWithCache : public ISubwordEncoder {
public:
	void Encode( const CString& word, CArray<int>& tokenIds,
		CArray<int>& tokenLengths ) const override final;

	// Cache cleanup period
	// The cache is used for Encode calls acceleration.
	// The result of the encode call is cached and will be erased if 
	// no call with the same word will occur among next 1-2 X cachePeriod calls.
	int GetCachePeriod() const { return cache.GetCachePeriod(); }

	// Sets the cache cleanup period.
	// Increase in cachePeriod leads to a in increase in memory consumption.
	// To completely switch the cache off set cachePeriod equal to -1.
	// Value 0 is treated as invalid.
	void SetCachePeriod( int cachePeriod ) const { cache.SetCachePeriod( cachePeriod ); }

	// Clears cache.
	void ClearCache() const { cache.Clear(); }

protected:
	// 'Internal' Encode with the same meaning.
	virtual void DoEncode( const CString& word, CArray<int>& tokenIds,
		CArray<int>& tokenLengths ) const = 0;

private:
	// Internal cache for encoding requests.
	class CCache {
	public:
		CCache() : cacheTime( 0 ), cachePeriod( 50000 ) {}
		// Cache cleanup period
		int GetCachePeriod() const { return cachePeriod; }
		// Sets the cache cleanup period
		void SetCachePeriod( int newPeriod );
		// Requests data from cache.
		bool Request( const CString& word, CArray<int>& tokenIds,
			CArray<int>& tokenLengths );
		// Adds data to cache.
		void Add( const CString& word, const CArray<int>& tokenIds,
			const CArray<int>& tokenLengths );
		// Clears cache.
		void Clear() { cacheTime = 0; wordCache.DeleteAll(); }

	private:
		// Data stored in cache: token ids and their unicode lengths and the latest request time.
		struct CCachedData {
			CFastArray<int, 4> TokenIds;
			CFastArray<int, 4> TokenLengths;
			long long Time;

			CCachedData() : Time( 0 ) {}
			CCachedData( const CCachedData& other );
			CCachedData( CCachedData&& other );
		};

		// Current cache state.
		CMap<CString, CCachedData> wordCache;
		// Current cache time.
		long long cacheTime;
		// Cache cleanup period.
		int cachePeriod;
	};

	// Cache for Encode calls.
	mutable CCache cache;
};

DECLARE_NEOML_MODEL_NAME( BytePairEncoderModelName, "NeoMLBytePairEncoderModel" )

// Subword encoder using Byte-Pair-Encoding algorithm
class NEOML_API IBytePairEncoder : public ISubwordEncoderWithCache {
public:
	// A list of unique tokens ordered by order of merges during training (this is used when encoding).
	// The Id of a token in encoded words is <Id in this array> + GetUnknownTokenId() + 1
	using CBPEDictionary = CArray<CString>;

	// Initializes the encoder. Can be safely used only once.
	// Every token except the letters (or bytes), EOW and SOW must be a concatenation of two other tokens.
	// If not empty, EOW and SOW must be contained in 'tokens' exactly only once as a separate token.
	virtual void Initialize( const CBPEDictionary& tokens, const CParams& ) = 0;
	// Whether the encoder is ready or it needs to be initialized using Initialize() or Serialize()
	virtual bool IsInitialized() const = 0;
};

DECLARE_NEOML_MODEL_NAME( UnigramEncoderModelName, "NeoMLUnigramEncoderModel" )

// Subword encoder using Unigram algorithm
class NEOML_API IUnigramEncoder : public ISubwordEncoderWithCache {
public:
	// Unigram vocabulary entry
	struct CSubword {
		CSubword() = default;
		CSubword( CString text, double score ) : Text( std::move( text ) ), Score( score ) {}

		CString Text;
		// Considered as log-probability of this subword
		double Score = 0.0;
	};

	// A list of unique tokens
	using CUnigramDictionary = CArray<CSubword>;

	// Initializes the encoder. Can be safely used only once.
	virtual void Initialize( const CUnigramDictionary& tokens, const CParams& ) = 0;
	// Whether the encoder is ready or it needs to be initialized using Initialize() or Serialize()
	virtual bool IsInitialized() const = 0;

	// Returns a list of subtokens with their scores
	virtual void GetDictionary( CUnigramDictionary& tokens ) const = 0;
};

///////////////////////////////////////////////////////////////////////////////

inline ISubwordEncoder::CParams::CParams( CString endOfWordToken, CString startOfWordToken, bool useRawBytes, int unknownTokenId ) :
	EndOfWordToken( std::move( endOfWordToken ) ),
	StartOfWordToken( std::move( startOfWordToken ) ),
	UseRawBytes( useRawBytes ),
	UnknownTokenId( unknownTokenId )
{}

} // namespace NeoML
