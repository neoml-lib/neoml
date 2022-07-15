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
#include <NeoML/TraditionalML/Model.h>
#include <NeoML/TraditionalML/WordDictionary.h>

namespace NeoML {

// Subword encoder interface.
class NEOML_API ISubwordEncoder : virtual public IObject {
public:
	virtual ~ISubwordEncoder() override = default;

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

class NEOML_API IBytePairEncoder : public ISubwordEncoderWithCache {
public:
	// Returns encoder flags.
	virtual bool UseEndOfWordToken() const = 0;
	virtual bool UseStartOfWordToken() const = 0;

	// Initializes the encoder. Can be safely used only once.
	// Every token except the letters must be a concatenation of two smaller tokens.
	// Start-of-Word and End-of-Word are automatically added to the input word when encoding.
	// If not empty, startOfWordToken and endOfWordToken must be contained in 'tokens' exactly only once as a separate token.
	// As a part of longer tokens, startOfWordToken can be located only in the beginning,
	// endOfWordToken can be located only in the end of a token
	virtual void LoadDictionary( const CWordDictionary& tokens, 
		const CString& endOfWordToken, const CString& startOfWordToken ) = 0;

	// Returns the BPE vocabulary. The preservation of the original frequencies and EoW/BoW symbols is not guaranteed.
	// If End-of-Word and Start-of-Word are disabled, the parameter values are ignored.
	virtual void GetDictionary( CWordDictionary& tokens, 
		const CString& endOfWordToken = "</s>", const CString& startOfWordToken = "<s>" ) const = 0;
};

} // namespace NeoML
