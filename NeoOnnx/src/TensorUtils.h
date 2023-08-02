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

#include <limits>

#include <NeoML/NeoML.h>

#include "onnx.pb.h"

#include "NeoOnnxCheck.h"
#include "Tensor.h"

namespace NeoOnnx {

template<class T>
constexpr const T& Clamp( const T& value, const T& low, const T& high )
{
	return value < low ? low : ( high < value ? high : value );
}

// Checks if float contains an integer value
bool IsInteger( float x );

// Auxiliary tensor's data loading functions

// Gets NeoML blob type from onnx tensor's data type
TBlobType GetBlobType( const onnx::TensorProto_DataType& onnxDataType );

// Loads data from raw bytes as an array of TSrc and stores it as an array of TDst (via static_cast)
template<class TSrc, class TDst>
inline void LoadFromRawData( const std::string& rawSrc, TDst* dest )
{
	const TSrc* src = reinterpret_cast<const TSrc*>( rawSrc.data() );
	for( size_t i = 0; i < rawSrc.size() / sizeof( TSrc ); ++i ) {
		TSrc value = Clamp( src[i], static_cast<TSrc>( std::numeric_limits<TDst>::lowest() ),
			static_cast<TSrc>( (std::numeric_limits<TDst>::max)() ) );
		dest[i] = static_cast<TDst>( value );
	}
}

// Loads NeoML's blob data (of type T) from onnx::TensorProto
template<class T>
inline void LoadBlobData( const onnx::TensorProto& src, CDnnBlob& dest )
{
	dest.Clear();
	T* buffer = dest.GetBuffer<T>( 0, dest.GetDataSize(), false );
	const bool isRaw = src.has_raw_data();
	switch( src.data_type() ) {
		case onnx::TensorProto::FLOAT:
			if( isRaw ) {
				LoadFromRawData<float, T>( src.raw_data(), buffer );
			} else {
				for( int valueIndex = 0; valueIndex < src.float_data_size(); ++valueIndex ) {
					buffer[valueIndex] = static_cast<T>( src.float_data( valueIndex ) );
				}
			}
			break;
		case onnx::TensorProto::DOUBLE:
			// Here downcast may happen (double -> float)
			if( isRaw ) {
				LoadFromRawData<double, T>( src.raw_data(), buffer );
			} else {
				for( int valueIndex = 0; valueIndex < src.double_data_size(); ++valueIndex ) {
					buffer[valueIndex] = static_cast<T>( src.double_data( valueIndex ) );
				}
			}
			break;
		case onnx::TensorProto::BOOL:
		case onnx::TensorProto::INT8:
		case onnx::TensorProto::UINT8:
		case onnx::TensorProto::INT16:
		case onnx::TensorProto::UINT16:
		case onnx::TensorProto::INT32:
			if( isRaw ) {
				LoadFromRawData<int, T>( src.raw_data(), buffer );
			} else {
				for( int valueIndex = 0; valueIndex < src.int32_data_size(); ++valueIndex ) {
					buffer[valueIndex] = static_cast<T>( src.int32_data( valueIndex ) );
				}
			}
			break;
		// NeoML works only with 32-bit signed types
		// But sometimes data values are quite small (like some indices)
		// that's why we can try to load them into NeoML data type
		case onnx::TensorProto::UINT32:
		case onnx::TensorProto::UINT64:
			if( isRaw ) {
				LoadFromRawData<uint64_t, T>( src.raw_data(), buffer );
			} else {
				for( int valueIndex = 0; valueIndex < src.uint64_data_size(); ++valueIndex ) {
					uint64_t value = Clamp( static_cast<uint64_t>( src.uint64_data( valueIndex ) ),
						static_cast<uint64_t>( 0 ), static_cast<uint64_t>( (std::numeric_limits<T>::max)() ) );
					buffer[valueIndex] = static_cast<T>( value );
				}
			}
			break;
		case onnx::TensorProto::INT64:
			if( isRaw ) {
				LoadFromRawData<int64_t, T>( src.raw_data(), buffer );
			} else {
				for( int valueIndex = 0; valueIndex < src.int64_data_size(); ++valueIndex ) {
					int64_t value = Clamp( src.int64_data( valueIndex ),
						static_cast<int64_t>( std::numeric_limits<T>::lowest() ),
						static_cast<int64_t>( (std::numeric_limits<T>::max)() ) );
					buffer[valueIndex] = static_cast<T>( value );
				}
			}
			break;
		case onnx::TensorProto::FLOAT16:
		case onnx::TensorProto::BFLOAT16:
		case onnx::TensorProto::COMPLEX64:
		case onnx::TensorProto::COMPLEX128:
		case onnx::TensorProto::UNDEFINED:
		default:
			dest.ReleaseBuffer( buffer, false );
			CheckNeoOnnxSupport( false, "tensor type" );
	}
	dest.ReleaseBuffer( buffer, true );
}

//---------------------------------------------------------------------------------------------------------------------
// Auxiliary tensor layout functions

// Converts tensor to the given layout
CPtr<const CTensorBase> ConvertTensor( const CTensorBase& inputTensor, const CTensorLayout& destLayout );
CPtr<const CUserTensor> ConvertTensor( const CUserTensor& inputTensor, const CTensorLayout& destLayout );
CPtr<const CDataTensor> ConvertTensor( const CDataTensor& inputTensor, const CTensorLayout& destLayout );
CPtr<const CShapeTensor> ConvertTensor( const CShapeTensor& inputTensor, const CTensorLayout& destLayout );

//---------------------------------------------------------------------------------------------------------------------
// Auxiliary tensor padding functions

// Calculates padding size if autoPad is SAME_UPPER or SAME_LOWER
void CalculatePadding( const CString& autoPad, const CTensorShape& kernelShape, CFastArray<int, 8>& pads );

// Pads user tensor via padding mode
// Last pads.Size() / 2 dimensions of the input tensor will be padded (it's compatible with both Conv and Pad onnx operators)
// First pads.Size() / 2 numbers determine padding size at the front of the dims
// Last pads.Size() / 2 numbers determine padding size at the back of the dims
CPtr<const CUserTensor> PadUserTensor( const CUserTensor& input, const CFastArray<int, 8>& pads,
	TBlobResizePadding padding, float padValue );

//---------------------------------------------------------------------------------------------------------------------
// Auxiliary tensor broadcast functions

// Tensor broadcast types
enum TBroadcastType {
	// Broadcast not supported
	BT_None,
	// Onnx custom broadcast, used in some older opset versions
	BT_Onnx,
	// Numpy-style broadcast, used in later opset versions
	BT_Numpy,
	// Upsample operator style
	// Number of dimensions must match
	// Supports broadcasting of non-trivial dimensions (size != 1)
	BT_Upsample,

	BT_Count
};

// Broadcast info
struct CBroadcast {
	// Broadcast type
	TBroadcastType Type;
	// Broadcasted axis index (used in BT_Onnx broadcast)
	int Axis;

	explicit CBroadcast( TBroadcastType type, int axis = NotFound ) : Type( type ), Axis( axis ) {}
};

// Calculates shape of the result of the broadcast operation
// If shapes can be broadcasted then it writes broadcasted shape to the result and returns true
// Returns false if shapes can't be broadcasted (in this case result will be empty)
bool BroadcastTensorShape( const CTensorShape& first, const CTensorShape& second, const CBroadcast& broadcast,
	CTensorShape& result );

// Prepares user tensor for CBroadcastLayer
CPtr<const CTensorBase> PrepareForBroadcast( const CTensorBase& input, const CBroadcast& broadcast, int outputDims );

//---------------------------------------------------------------------------------------------------------------------

// Converts the given tensor to user tensor by adding corresponding data layer to the dnn (if needed)
CPtr<const CUserTensor> AsUserTensor( const CTensorBase& tensor, const CString& layerName, CDnn& dnn );

// Converts the given tensor to shape tensor by adding corresponding layers to the dnn (if needed)
CPtr<const CShapeTensor> AsShapeTensor( const CTensorBase& tensor, const CString& layerName, CDnn& dnn );

// Converts the given array to shape tensor by adding corresponding layers to the dnn (if needed)
CPtr<const CShapeTensor> AsShapeTensor( const CFastArray<int, 8>& data, const CString& layerName, CDnn& dnn );
CPtr<const CShapeTensor> AsShapeTensor( const CFastArray<float, 8>& data, const CString& layerName, CDnn& dnn );

//---------------------------------------------------------------------------------------------------------------------

// Extracts shape to the given array
// Throws an exception if CUserTensor is provided (CUserTensor doesn't have shape)
void GetTensorShape( const CTensorBase& tensor, CTensorShape& shape );

//---------------------------------------------------------------------------------------------------------------------

// Information about renaming (change dimension names without reordering them in memory)
struct CTensorLayoutRename {
	CTensorLayout From;
	CTensorLayout To;
};

// Information about swapping
struct CTensorLayoutTranspose {
	CTensorLayoutTranspose( TBlobDim first, TBlobDim second ) : First( first ), Second( second ) {}

	TBlobDim First;
	TBlobDim Second;
};

inline int TensorLayoutHash( const CTensorLayout& layout )
{
	static_assert( static_cast<int>( BD_Count ) <= 8, "BD_Count > 8" );
	int result = 0;
	for( int i = 0; i < layout.Size(); ++i ) {
		result |= static_cast<int>( layout[i] ) << ( 3 * i );
	}
	return result;
}

struct CTensorLayoutMatchFunctor {
	explicit CTensorLayoutMatchFunctor( const CTensorLayout& layout ) : layout( layout ) {}
	bool operator()( const CTensorLayout& candidate ) const { return layout == candidate; }

	void Print() const { for( TBlobDim dim : layout ) std::cout << "\t" << (int)dim; }

private:
	CTensorLayout layout;
};

// Finds optimal way to convert inputLayout into layout where TFunctor()( layout ) == true
template<typename CFunctor>
CTensorLayout FindOptimalConversion( const CTensorLayout& inputLayout, const CFunctor& functor,
	CTensorLayoutRename& renameBeforeTransposes, CFastArray<CTensorLayoutTranspose, 2>& transposes,
	CTensorLayoutRename& renameAfterTransposes )
{
	renameBeforeTransposes = CTensorLayoutRename();
	transposes.Empty();
	renameAfterTransposes = CTensorLayoutRename();

	if( functor( inputLayout ) ) {
		return inputLayout;
	}

	struct CBfsEntry {
		CBfsEntry() = default;
		CBfsEntry( const CBfsEntry& other ) :
			Rename( other.Rename ),
			OutputLayout( other.OutputLayout )
		{
			other.Transposes.CopyTo( Transposes );
		};
		CBfsEntry( CBfsEntry&& other )
		{
			other.Rename.From.MoveTo( Rename.From );
			other.Rename.To.MoveTo( Rename.To );
			other.Transposes.MoveTo( Transposes );
			other.OutputLayout.MoveTo( OutputLayout );
		};

		CTensorLayoutRename Rename;
		CFastArray<CTensorLayoutTranspose, 2> Transposes;
		CTensorLayout OutputLayout;
	};

	CArray<CBfsEntry> queue;
	CHashTable<int> visited;
	visited.Add( TensorLayoutHash( inputLayout ) );

	queue.SetSize( 1 );
	queue[0].OutputLayout = inputLayout;

	// Add all renamings to the queue
	CTensorLayoutRename currentRename;
	CTensorLayout currentOutputLayout;
	CTensorLayout currentInputLayout = inputLayout;
	currentRename.From = inputLayout;
	currentRename.From.QuickSort<Ascending<TBlobDim>>();
	currentRename.To.SetSize( currentRename.From.Size() );
	currentOutputLayout.SetSize( inputLayout.Size() );

	auto bruteForceRename = [&] ( int sortedAxisIndex, bool isPreTranspose, auto&& bruteForceRename ) -> bool {
		static_assert( static_cast<int>( BD_Count ) == 7, "BD_Count != 7" );
		const bool isLastAxis = sortedAxisIndex == currentOutputLayout.Size() - 1;
		const int axisIndex = currentInputLayout.Find( currentRename.From[sortedAxisIndex] );
		const int minValue = sortedAxisIndex == 0 ? 0 : static_cast<int>( currentRename.To[sortedAxisIndex - 1] ) + 1;
		const int maxValue = static_cast<int>( BD_Count ) - ( currentOutputLayout.Size() - sortedAxisIndex );
		for( int value = minValue; value <= maxValue; ++value ) {
			currentRename.To[sortedAxisIndex] = static_cast<TBlobDim>( value );
			currentOutputLayout[axisIndex] = static_cast<TBlobDim>( value );
			if( isLastAxis ) {
				if( functor( currentOutputLayout ) ) {
					( isPreTranspose ? renameBeforeTransposes : renameAfterTransposes ) = currentRename;
					return true;
				}
				const int hash = isPreTranspose ? TensorLayoutHash( currentOutputLayout ) : 0;
				if( isPreTranspose && !visited.Has( hash ) ) {
					visited.Add( hash );
					CBfsEntry newEntry;
					newEntry.Rename = currentRename;
					newEntry.OutputLayout = currentOutputLayout;
					queue.Add( newEntry );
				}
			} else if( bruteForceRename( sortedAxisIndex + 1, isPreTranspose, bruteForceRename ) ) {
				return true;
			}
		}
		return false;
	};

	if( bruteForceRename( 0, true, bruteForceRename ) ) {
		return currentOutputLayout;
	}

	NeoAssert( inputLayout.Size() > 1 ); // layout of size must be converted via rename
	// std::cout << "Queue size after renames: " << queue.Size() << '\n';

	int queueIndex = 0;
	const int queueResetPeriod = 1024;

	auto tryAddTranspose = [&] ( const CBfsEntry& entry, const CTensorLayoutTranspose& transpose ) -> bool {
		const int firstIndex = entry.OutputLayout.Find( transpose.First );
		const int secondIndex = entry.OutputLayout.Find( transpose.Second );
		currentOutputLayout = entry.OutputLayout;
		if( firstIndex == NotFound && secondIndex == NotFound ) {
			return false;
		} else if( firstIndex == NotFound ) {
			currentOutputLayout[secondIndex] = transpose.First;
		} else if( secondIndex == NotFound ) {
			currentOutputLayout[firstIndex] =  transpose.Second;
		} else {
			std::swap( currentOutputLayout[firstIndex], currentOutputLayout[secondIndex] );
		}

		if( functor( currentOutputLayout ) ) {
			entry.Transposes.CopyTo( transposes );
			transposes.Add( transpose );
			renameBeforeTransposes = entry.Rename;
			return true;
		}

		const int hash = TensorLayoutHash( currentOutputLayout );
		if( !visited.Has( hash ) ) {
			CBfsEntry newEntry( entry );
			newEntry.OutputLayout = currentOutputLayout;
			newEntry.Transposes.Add( transpose );
			queue.Add( newEntry );
			visited.Add( hash );
		}

		currentRename.From = currentOutputLayout;
		currentRename.From.QuickSort<Ascending<TBlobDim>>();
		currentInputLayout = currentOutputLayout;
		if( bruteForceRename( 0, false, bruteForceRename ) ) {
			entry.Transposes.CopyTo( transposes );
			transposes.Add( transpose );
			renameBeforeTransposes = entry.Rename;
			return true;
		}

		return false;
	};

	int currStepQueueSize = queue.Size();
	int step = 0;
	while( queueIndex < queue.Size() ) {
		if( queueIndex == queueResetPeriod ) {
			queueIndex = 0;
			currStepQueueSize -= queueResetPeriod; 
			queue.DeleteAt( 0, queueResetPeriod );
		}

		const CBfsEntry entry = queue[queueIndex];

		TBlobDim secondLastAxis = BD_BatchLength;
		TBlobDim lastAxis = BD_BatchLength;
		for( TBlobDim dim : entry.OutputLayout ) {
			if( dim > lastAxis ) {
				secondLastAxis = lastAxis;
				lastAxis = dim;
			} else if( dim > secondLastAxis ) {
				secondLastAxis = dim;
			}
		}

		// Add all plain transposes
		// 1. Transpose last 2 axes
		CTensorLayoutTranspose transpose( secondLastAxis, lastAxis );
		if( tryAddTranspose( entry, transpose ) ) {
			return currentOutputLayout;
		}

		// 2. Transpose non-last used axis with unused axis after last axis
		for( TBlobDim first : entry.OutputLayout ) {
			transpose.First = first;
			if( transpose.First == lastAxis ) {
				continue;
			}
			for( transpose.Second = lastAxis + 1; transpose.Second != BD_Count; ++transpose.Second ) {
				if( tryAddTranspose( entry, transpose ) ) {
					return currentOutputLayout;
				}
			}
		}

		// 3. Transpose last used axis with unused axis less than last axis
		transpose.Second = lastAxis;
		for( transpose.First = BD_BatchLength; transpose.First != transpose.Second; ++transpose.First ) {
			if( entry.OutputLayout.Find( transpose.First ) != NotFound ) {
				continue;
			}
			if( tryAddTranspose( entry, transpose ) ) {
				return currentOutputLayout;
			}
		}

		// Add all other possible transposes
		for( transpose.First = BD_BatchLength; transpose.First != BD_Channels; ++transpose.First ) {
			for( transpose.Second = transpose.First + 1; transpose.Second != BD_Count; ++transpose.Second ) {
				if( tryAddTranspose( entry, transpose ) ) {
					return currentOutputLayout;
				}
			}
		}

		queueIndex++;
		if( queueIndex == currStepQueueSize ) {
			currStepQueueSize = queue.Size();
			++step;
			std::cout << "Failed to find a way in " << step << " transposes\n";
			std::cout << "\tFrom:";
			for( TBlobDim dim : inputLayout ) {
				std::cout << '\t' << (int ) dim;
			}
			std::cout << "\n\tTo:";
			functor.Print();
			std::cout << '\n';
		}
	}

	NeoAssert( false );
	return inputLayout;
}

} // namespace NeoOnnx

