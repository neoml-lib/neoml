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
// Layout conversion functions

class ITensorLayoutValidator {
public:
	virtual ~ITensorLayoutValidator() = default;

	// Returns whether the given layout satisfies the criteria
	virtual bool operator()( const CTensorLayout& layout ) const = 0;
};

// Converts tensor to the layout accepted by validator
CPtr<const CTensorBase> ConvertTensor( const CTensorBase& inputTensor, const ITensorLayoutValidator& validator );
CPtr<const CUserTensor> ConvertTensor( const CUserTensor& inputTensor, const ITensorLayoutValidator& validator );

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

// Generates layout recommended for broadcasting given tensor
CTensorLayout BroadcastTensorLayout( const CTensorLayout& inputLayout, const CBroadcast& broadcast, int outputDims );

// Prepares tensor for operation with broadcast
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
// Auxiliary tensor layout functions

// Information about renaming (change dimension names without reordering them in memory)
struct CTensorLayoutRename {
	CTensorLayout From;
	CTensorLayout To;
};

// Information about transposition (swapping 2 dimensions)
struct CTensorLayoutTranspose {
	CTensorLayoutTranspose( TBlobDim first, TBlobDim second ) : First( first ), Second( second ) {}

	TBlobDim First;
	TBlobDim Second;
};

// Information about layout conversion
// The chain is
//     -> [PreTransposeRename] -> [Transpose]* -> [PostTransposeRename] ->
// (each part is optional, trivial conversion means no operations at all)
struct CTensorLayoutConversion {
	CTensorLayoutRename PreTransposeRename;
	CFastArray<CTensorLayoutTranspose, 2> Transposes;
	CTensorLayoutRename PostTransposeRename;

	CTensorLayoutConversion() = default;
	~CTensorLayoutConversion() = default;
	CTensorLayoutConversion( const CTensorLayoutConversion& other );
	CTensorLayoutConversion( CTensorLayoutConversion&& other );
	CTensorLayoutConversion& operator=( const CTensorLayoutConversion& other );
	CTensorLayoutConversion& operator=( CTensorLayoutConversion&& other );
};

inline CTensorLayoutConversion::CTensorLayoutConversion( const CTensorLayoutConversion& other ) :
	PreTransposeRename( other.PreTransposeRename ),
	PostTransposeRename( other.PostTransposeRename )
{
	other.Transposes.CopyTo( Transposes );
}

inline CTensorLayoutConversion::CTensorLayoutConversion( CTensorLayoutConversion&& other )
{
	other.PreTransposeRename.From.MoveTo( PreTransposeRename.From );
	other.PreTransposeRename.To.MoveTo( PreTransposeRename.To );
	other.Transposes.MoveTo( Transposes );
	other.PostTransposeRename.From.MoveTo( PostTransposeRename.From );
	other.PostTransposeRename.To.MoveTo( PostTransposeRename.To );
}

inline CTensorLayoutConversion& CTensorLayoutConversion::operator=( const CTensorLayoutConversion& other )
{
	PreTransposeRename = other.PreTransposeRename;
	other.Transposes.CopyTo( Transposes );
	PostTransposeRename = other.PostTransposeRename;
	return *this;
}

inline CTensorLayoutConversion& CTensorLayoutConversion::operator=( CTensorLayoutConversion&& other )
{
	other.PreTransposeRename.From.MoveTo( PreTransposeRename.From );
	other.PreTransposeRename.To.MoveTo( PreTransposeRename.To );
	other.Transposes.MoveTo( Transposes );
	other.PostTransposeRename.From.MoveTo( PostTransposeRename.From );
	other.PostTransposeRename.To.MoveTo( PostTransposeRename.To );
	return *this;
}

// Finds optimal way to convert inputLayout into a valid layout ( validator( layout ) == true )
CTensorLayout FindConversion( const CTensorLayout& inputLayout, const ITensorLayoutValidator& validator,
	CTensorLayoutConversion& conversion );

//---------------------------------------------------------------------------------------------------------------------
// Implementations of ITensorLayoutValidator

// Validator which considers as a valid only etalon layout
class CTensorLayoutMatchValidator : public ITensorLayoutValidator {
public:
	explicit CTensorLayoutMatchValidator( const CTensorLayout& etalon ) : etalon( etalon ) {}

	bool operator()( const CTensorLayout& layout ) const override { return etalon == layout; }

private:
	CTensorLayout etalon;
};

// Validator which considers any ONNX-compatible layout as valid
class COnnxTensorLayoutValidator : public ITensorLayoutValidator {
public:
	bool operator()( const CTensorLayout& layout ) const override { return !IsTransposedLayout( layout ); }
};

// Validator which considers NeoML-compatible image layouts as valid
class CNeoMLImageLayoutValidator : public ITensorLayoutValidator {
public:
	bool operator()( const CTensorLayout& layout ) const override;
};

inline bool CNeoMLImageLayoutValidator::operator()( const CTensorLayout& layout ) const
{
	if( ( !layout.IsEmpty() && layout[0] >= BD_Height ) || ( layout.Size() > 1 && layout[1] != BD_Channels ) ) {
		return false;
	}

	for( int i = 2; i < layout.Size(); ++i ) {
		if( layout[i] != BD_Height + ( i - 2 ) ) {
			return false;
		}
	}

	return true;
}

// Validator for batch normalization operator
class CBatchNormLayoutValidator : public ITensorLayoutValidator {
public:
	bool operator()( const CTensorLayout& layout ) const override;
};

inline bool CBatchNormLayoutValidator::operator()( const CTensorLayout& layout ) const
{
	if( ( !layout.IsEmpty() && layout[0] >= BD_Height ) || ( layout.Size() > 1 && layout[1] != BD_Channels ) ) {
		return false;
	}

	for( int i = 2; i < layout.Size(); ++i ) {
		if( layout[i] < BD_Height || layout[i] == BD_Channels ) {
			return false;
		}
	}

	return true;
}

} // namespace NeoOnnx

