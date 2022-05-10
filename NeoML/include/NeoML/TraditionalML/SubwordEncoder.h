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

namespace NeoML {

// Subword encoder interface.
class NEOML_API ISubwordEncoder : virtual public IObject {
public:
	~ISubwordEncoder() override = default;

	// Encodes a word as a sequence of token ids with corresponding token lengths.
	// TokenId range = [minId, ... , maxId].
	virtual void Encode( const CString& word, CArray<int>& tokenIds,
		CArray<int>& tokenLengths ) const = 0;
	
	// Decodes sequence of token ids into a sequence of words.
	virtual void Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const = 0;

	// Returns token id range.
	virtual void GetTokenIdRange( int& minId, int& maxId ) const = 0;
};

// Subword encoder which supports caching results of 'Encode' calls.
class NEOML_API ISubwordEncoderWithCache : public ISubwordEncoder {
public:
	virtual void Encode( const CString& word, CArray<int>& tokenIds,
		CArray<int>& tokenLengths ) const override final;

	// Sets the cache cleanup period.
	// The cache is used for Encode calls acceleration.
	// The result of the encode call is cached and will be erased if 
	// no call with the same word will occur among next 1-2 X cachePeriod calls.
	// Increase in cachePeriod leads to a in increase in memory consumption.
	// To completely switch the cache off set cachePeriod equal to -1.
	// Value 0 is treated as invalid.
	void SetCachePeriod( int cachePeriod ) const { cache.SetCachePeriod( cachePeriod ); }

protected:
	// Encodes a word as a sequence of token ids with corresponding token lengths.
	// TokenId range = [0, ... , Size() - 1].
	virtual void doEncode( const CString& word, CArray<int>& tokenIds,
		CArray<int>& tokenLengths ) const = 0;

private:
	// Internal cache for encoding requests.
	class CCache {
	public:
		CCache();
		// Sets the cache cleanup period
		void SetCachePeriod( int newPeriod );
		// Requests data from cache.
		bool Request( const CString& word, CArray<int>& tokenIds,
			CArray<int>& tokenLengths );
		// Adds data to cache.
		void Add( const CString& word, const CArray<int>& tokenIds,
			const CArray<int>& tokenLengths );

	private:
		// Data stored in cache: token ids and their uniode lengths and the lattest request time.
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

class NEOML_API IBytePairEncoder : public ISubwordEncoderWithCache {
public:
	// Returns encoder flags.
	virtual bool UseEndOfWordToken() const = 0;
	virtual bool UseStartOfWordToken() const = 0;
};

} // namespace NeoML
