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

#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/SubwordEncoder.h>

namespace NeoML {

static constexpr int SubwordEncoderParamsCurrentVersion = 0;

void ISubwordEncoder::CParams::Serialize( CArchive& archive )
{
	archive.SerializeVersion( SubwordEncoderParamsCurrentVersion );
	archive.Serialize( EndOfWordToken );
	archive.Serialize( StartOfWordToken );
	archive.Serialize( UseRawBytes );
	archive.Serialize( UnknownTokenId );
}

//////////////////////////////////////

ISubwordEncoderWithCache::CCache::CCachedData::CCachedData( const CCachedData& other ) :
	Time( other.Time )
{
	other.TokenIds.CopyTo( TokenIds );
	other.TokenLengths.CopyTo( TokenLengths );
}

ISubwordEncoderWithCache::CCache::CCachedData::CCachedData( CCachedData&& other ) :
	Time( other.Time )
{
	other.TokenIds.MoveTo( TokenIds );
	other.TokenLengths.MoveTo( TokenLengths );
}

///////////////////////////////////////////////////////////////////////////////

void ISubwordEncoderWithCache::CCache::SetCachePeriod( int newPeriod )
{
	NeoAssert( newPeriod == NotFound || newPeriod > 0 );
	cachePeriod = newPeriod;
	if( cachePeriod == NotFound ) {
		Clear();
	}
}

bool ISubwordEncoderWithCache::CCache::Request( const CString& word,
	CArray<int>& tokenIds, CArray<int>& tokenLengths )
{
	if( cachePeriod == NotFound ) {
		return false;
	}

	cacheTime++;
	bool success = false;
	if( wordCache.Has( word ) ) {
		CCachedData& wordData = wordCache.Get( word );
		tokenIds.SetBufferSize( wordData.TokenIds.Size() );
		tokenLengths.SetBufferSize( wordData.TokenLengths.Size() );

		for( int i = 0; i < wordData.TokenIds.Size(); i++ ) {
			tokenIds.Add( wordData.TokenIds[i] );
			tokenLengths.Add( wordData.TokenLengths[i] );
		}
		wordData.Time = cacheTime;
		success = true;
	}

	// Removes items from cache.
	// The item is erased if there has not been a request with the same word since the previous cleanup. 
	if( cacheTime % cachePeriod == 0 ) {
		CArray<CString> wordsToDelete;
		for( TMapPosition pos = wordCache.GetFirstPosition(); pos != NotFound;
			pos = wordCache.GetNextPosition( pos ) )
		{
			const CCachedData& wordData = wordCache.GetValue( pos );
			if( cacheTime - wordData.Time >= cachePeriod ) {
				wordsToDelete.Add( wordCache.GetKey( pos ) );
			}
		}

		for( int i = 0; i < wordsToDelete.Size(); i++ ) {
			wordCache.Delete( wordsToDelete[i] );
		}
	}

	return success;
}

void ISubwordEncoderWithCache::CCache::Add( const CString& word,
	const CArray<int>& tokenIds, const CArray<int>& tokenLengths )
{
	// If cache is disabled
	if( cachePeriod == NotFound ) {
		return;
	}

	NeoAssert( !wordCache.Has( word ) );
	NeoAssert( tokenIds.Size() == tokenLengths.Size() );

	CCachedData wordData;
	wordData.Time = cacheTime;
	for( int i = 0; i < tokenIds.Size(); i++ ) {
		wordData.TokenIds.Add( tokenIds[i] );
		wordData.TokenLengths.Add( tokenLengths[i] );
	}
	wordCache.Add( word, wordData );
}

///////////////////////////////////////////////////////////////////////////////

void ISubwordEncoderWithCache::Encode( const CString& word, CArray<int>& tokenIds,
	CArray<int>& tokenLengths ) const
{
	if( cache.Request( word, tokenIds, tokenLengths ) ) {
		return;
	}

	CArray<int> wordTokenIds;
	CArray<int> wordTokenLengths;
	DoEncode( word, wordTokenIds, wordTokenLengths );

	tokenIds.Add( wordTokenIds );
	tokenLengths.Add( wordTokenLengths );

	cache.Add( word, wordTokenIds, wordTokenLengths );
}
} // namespace NeoML
