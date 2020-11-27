/* Copyright © 2017-2020 ABBYY Production LLC

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

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// 2D data private to each OMP block thread
class COmpPrivate2DData : public CCrtAllocatedObject {
public:
	explicit COmpPrivate2DData( int threadCount, IMathEngine& mathEngine, int _height, int _width ) :
		count( threadCount ),
		height( _height ),
		width( _width ),
		dataSize( _height * _width ),
		items( 0 )
	{
		items = (CFloatHandleVar*) malloc( count * sizeof(CFloatHandleVar) );
		CFloatHandleVar* ptr = items;
		for( int i = 0; i < count; i++ ) {
			::new(ptr) CFloatHandleVar( mathEngine, dataSize );
			ptr++;
		}
	}

	~COmpPrivate2DData()
	{
		CFloatHandleVar* ptr = items;
		for( int i = 0; i < count; i++ ) {
			ptr->~CFloatHandleVar();
			ptr++;
		}
		free( items );
	}

	// Get the data for the current thread
	CFloatHandle GetPrivateData() const
	{
		const int threadNumber = OmpGetThreadNum();
		return items[threadNumber].GetHandle();
	}

	// Get the data height
	int GetHeight() const { return height; }

	// Get the data width
	int GetWidth() const { return width; }

	// Get the total data size
	int GetDataSize() const { return dataSize; }

private:
	const int count; // the number of objects, same as the number of threads
	const int height; // the data height
	const int width; // the data width
	const int dataSize; // the total size
	CFloatHandleVar* items; // the objects for the rest of the threads
};

//------------------------------------------------------------------------------------------------------------

// 1D data private to each OMP block thread
class COmpPrivate1DData : public CCrtAllocatedObject {
public:
	COmpPrivate1DData( int threadCount, IMathEngine& mathEngine, int size ) :
		count( threadCount ),
		dataSize( size ),
		items( 0 )
	{
		items = (CFloatHandleVar*) malloc( count * sizeof(CFloatHandleVar) );
		CFloatHandleVar* ptr = items;
		for( int i = 0; i < count; i++ ) {
			::new(ptr) CFloatHandleVar( mathEngine, dataSize );
			ptr++;
		}
	}

	~COmpPrivate1DData()
	{
		CFloatHandleVar* ptr = items;
		for( int i = 0; i < count; i++ ) {
			ptr->~CFloatHandleVar();
			ptr++;
		}
		free( items );
	}

	// Get the data for the current thread
	CFloatHandle GetPrivateData() const
	{
		const int threadNumber = OmpGetThreadNum();
		return items[threadNumber].GetHandle();
	}

	// Get the data size
	int GetDataSize() const { return dataSize; }

private:
	const int count; // the number of objects, same as the number of threads
	const int dataSize; // the data size
	CFloatHandleVar* items; // the objects for the rest of the threads
};

//------------------------------------------------------------------------------------------------------------

// 1D data for reduction
class COmpReduction1DData : public CCrtAllocatedObject {
public:
	CFloatHandle Data;
	int Size;

	COmpReduction1DData( IMathEngine& _mathEngine, const CFloatHandle& data, int size ) :
		Data( data ),
		Size( size ),
		mathEngine( _mathEngine ),
		dataHolder( _mathEngine, 0 )
	{
	}

	COmpReduction1DData( const COmpReduction1DData& other );

	void Reduce( const COmpReduction1DData& other );

private:
	IMathEngine& mathEngine;
	CFloatHandleVar dataHolder;
};

inline COmpReduction1DData::COmpReduction1DData( const COmpReduction1DData& other ) :
	Size( other.Size ),
	mathEngine( other.mathEngine ),
	dataHolder( other.mathEngine, other.Size )
{
	mathEngine.VectorFill( dataHolder.GetHandle(), 0.0f, Size );
	Data = dataHolder.GetHandle();
}

inline void COmpReduction1DData::Reduce( const COmpReduction1DData& other )
{
	mathEngine.VectorAdd( Data, other.Data, Data, Size );
}

// COmpReduction is the template for assigning the data to each OMP thread and merging the threads results
// Each thread's data type is defined by TItem
// TItem should provide these methods:
// 1)
//		TItem::TItem( const TItem& sharedItem );
//			* creates a copy of shared data for each thread
// 2)
//		void TItem::Reduce( const TItem& privateItem );
//			* reduces a private object to a shared one

template<class TItem>
class COmpReduction : public CCrtAllocatedObject {
public:
	explicit COmpReduction( int threadCount, TItem& _sharedItem ) :
		count( threadCount - 1 ),
		sharedItem( _sharedItem ),
		items( 0 )
	{
		ASSERT_EXPR( count >= 0 );
		items = reinterpret_cast<TItem*>( malloc( count * sizeof(TItem) ) );
		TItem* ptr = items;
		for( int i = 0; i < count; i++ ) {
			::new(ptr)TItem( sharedItem );
			ptr++;
		}
	}

	~COmpReduction()
	{
		TItem* ptr = items;
		for( int i = 0; i < count; i++ ) {
			ptr->~TItem();
			ptr++;
		}
		free( items );
	}

	int GetThreadCount() const { return count; }

	TItem& GetPrivate() const
	{
		const int threadNumber = OmpGetThreadNum();
		ASSERT_EXPR( 0 <= threadNumber && threadNumber < count + 1 );

		if( threadNumber == 0 ) {
			return sharedItem;
		}
		return items[threadNumber - 1];
	}

	void Reduce()
	{
		for( int i = 0; i < count; i++ ) {
			sharedItem.Reduce( items[i] );
		}
	}

private:
	const int count; // the number of objects, same as the number of threads
	TItem& sharedItem; // the shared object to which the private object is reduced
	TItem* items; // the objects for the rest of the threads
};

} // namespace NeoML
