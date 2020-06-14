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

namespace FObj {

template<typename ARRAY, typename COMPARE>
class CPriorityQueue {
public:
	typedef ARRAY ArrayType;
	typedef COMPARE CompareType;
	typedef CPriorityQueue<ArrayType, CompareType> QueueType;
	typedef typename ARRAY::TElement ElementType;

	explicit CPriorityQueue( const CompareType& compare = CompareType() ) : data( compare ) {}

	int Size() const { return buffer().Size(); }
	bool IsEmpty() const { return Size() == 0; }
	void SetBufferSize( int bufferSize ) { buffer().SetBufferSize( bufferSize ); }
	const ArrayType& GetBuffer() const { return buffer(); }

	void Push( const ElementType& element );
	const ElementType& Peek() const { AssertFO( !IsEmpty() ); return buffer()[0]; }
	bool Pop( ElementType& element );
	bool Pop();
	void PopAndPush( const ElementType& element );
	void Reset() { buffer().DeleteAll(); }
	void Attach( ArrayType& elementArray );
	void Detach( ArrayType& elementArray );
	void DetachAndSort( ArrayType& elementArray );

	template<typename TARGET>
	void CopyTo( TARGET& elementArray ) const;
	template<typename TARGET>
	void CopyToAndSort( TARGET& elementArray ) const;

private:
	struct CData : public CompareType {
		ArrayType buffer;

		CData( const COMPARE& compare ) : CompareType( compare ) {}
	};

	CData data;

	ArrayType& buffer() { return data.buffer; }
	const ArrayType& buffer() const { return data.buffer; }
	const CompareType& compare() const { return data; }
	void siftUp( ElementType* buffer, int index, const ElementType& element ) const;
	static int getParentIndex( int index ) { PresumeFO( index > 0 ); return ( index - 1 ) / 2; }
	void pop();
	void siftDown( ElementType* buffer, const ElementType& element, int size ) const;
	void sortHeap( ElementType* buffer, int size ) const;
};

template<typename ARRAY, typename COMPARE>
void CPriorityQueue<ARRAY, COMPARE>::Push( const ElementType& element )
{
	if( IsEmpty() ) {
		buffer().Add( element );
	} else {
		int index = buffer().Size();
		buffer().SetSize( index + 1 );
		siftUp( buffer().GetPtr(), index, element );
	}
}

template<typename ARRAY, typename COMPARE>
void CPriorityQueue<ARRAY, COMPARE>::siftUp(
	ElementType* elementArray, int index, const ElementType& element ) const
{
	for(;;) {
		if( index == 0 ) {
			break;
		}

		int parentIndex = getParentIndex( index );
		const ElementType& parentElement = elementArray[parentIndex];
		if( !compare().Predicate( parentElement, element ) ) {
			break;
		}

		elementArray[index] = parentElement;
		index = parentIndex;
	}

	elementArray[index] = element;
}

template<typename ARRAY, typename COMPARE>
inline bool CPriorityQueue<ARRAY, COMPARE>::Pop( ElementType& element )
{
	if( IsEmpty() ) {
		return false;
	}

	element = buffer()[0];
	pop();
	return true;
}

template<typename ARRAY, typename COMPARE>
inline bool CPriorityQueue<ARRAY, COMPARE>::Pop()
{
	if( IsEmpty() ) {
		return false;
	}

	pop();
	return true;
}

template<typename ARRAY, typename COMPARE>
inline void CPriorityQueue<ARRAY, COMPARE>::pop()
{
	PresumeFO( !IsEmpty() );

	int lastIndex = buffer().Size() - 1;
	if( lastIndex > 0 ) {
		siftDown( buffer().GetPtr(), buffer()[lastIndex], lastIndex );
	}

	buffer().DeleteAt( lastIndex );
}

template<typename ARRAY, typename COMPARE>
void CPriorityQueue<ARRAY, COMPARE>::siftDown(
	ElementType* elementArray, const ElementType& element, int size ) const
{
	int parentIndex = 0;
	for(;;) {
		int childIndex = 2 * parentIndex + 1;
		if( childIndex >= size ) {
			break;
		}

		int otherChildIndex = childIndex + 1;
		if( otherChildIndex < size &&
			compare().Predicate( elementArray[childIndex], elementArray[otherChildIndex] ) )
		{
			childIndex = otherChildIndex;
		}

		if( !compare().Predicate( element, elementArray[childIndex] ) ) {
			break;
		}

		elementArray[parentIndex] = elementArray[childIndex];
		parentIndex = childIndex;
	}

	elementArray[parentIndex] = element;
}

template<typename ARRAY, typename COMPARE>
void CPriorityQueue<ARRAY, COMPARE>::PopAndPush( const ElementType& element )
{
	AssertFO( !IsEmpty() );
	siftDown( buffer().GetPtr(), element, Size() );
}

template<typename ARRAY, typename COMPARE>
void CPriorityQueue<ARRAY, COMPARE>::Attach( ArrayType& elementArray )
{
	for( int i = 1; i < elementArray.Size(); i++ ) {
		int parentIndex = getParentIndex( i );
		const ElementType& parentElement = elementArray[parentIndex];
		if( compare().Predicate( parentElement, elementArray[i] ) ) {
			ElementType element = elementArray[i];
			elementArray[i] = parentElement;
			siftUp( elementArray.GetPtr(), parentIndex, element );
		}
	}

	elementArray.MoveTo( buffer() );
}

template<typename ARRAY, typename COMPARE>
void CPriorityQueue<ARRAY, COMPARE>::Detach( ArrayType& elementArray )
{
	buffer().MoveTo( elementArray );
}

template<typename ARRAY, typename COMPARE>
void CPriorityQueue<ARRAY, COMPARE>::DetachAndSort( ArrayType& elementArray )
{
	buffer().MoveTo( elementArray );
	sortHeap( elementArray.GetPtr(), elementArray.Size() );
}

template<typename ARRAY, typename COMPARE>
void CPriorityQueue<ARRAY, COMPARE>::sortHeap( ElementType* buffer, int size ) const
{
	for( int i = size - 1; i > 0; i-- ) {
		ElementType root = buffer[0];
		siftDown( buffer, buffer[i], i );
		buffer[i] = root;
	}
}

template<typename ARRAY, typename COMPARE>
template<typename TARGET>
void CPriorityQueue<ARRAY, COMPARE>::CopyTo( TARGET& elementArray ) const
{
	elementArray.SetBufferSize( Size() );
	for( int i = 0; i < Size(); i++ ) {
		elementArray.Add( buffer()[i] );
	}
}

template<typename ARRAY, typename COMPARE>
template<typename TARGET>
void CPriorityQueue<ARRAY, COMPARE>::CopyToAndSort( TARGET& elementArray ) const
{
	CopyTo( elementArray );
	sortHeap( elementArray.GetPtr(), elementArray.Size() );
}

} // namespace FObj
