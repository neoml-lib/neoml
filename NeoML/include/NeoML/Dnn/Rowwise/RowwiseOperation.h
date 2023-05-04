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

namespace NeoML {

struct CRowwiseOperationDesc;
class CBlobDesc;
class IMathEngine;

// Interface of row-wise operation
// Row-wise operation is an operation during which one row of output image requires only a few rows of input image
// Activation or convolution can be calculated rowwise (when global pooling or object normalization can not)
class NEOML_API IRowwiseOperation : virtual public IObject {
public:
	~IRowwiseOperation() override;

	// Returns pointer to operation NeoMathEngine descriptor
	// The descriptor pointer is valid till next Reshape call (or till this object is destroyed)
	// The user must delete this pointer afterwards
	virtual CRowwiseOperationDesc* GetDesc( const CBlobDesc& inputDesc ) = 0;

	void Serialize( CArchive& archive ) override = 0;
};

// Registration macro
#define REGISTER_NEOML_ROWWISE_OPERATION( classType, name ) \
	static CRowwiseOperationRegistrar<classType> __merge__1( _RegisterRowwise, __LINE__ )( name );

// Get registered name from the object
NEOML_API const char* GetRowwiseOperationName( const IObject* rowwiseOperation );

// Create object of registered name
NEOML_API CPtr<IObject> CreateRowwiseOperation( const char* className, IMathEngine& mathEngine );

template<class T>
inline CPtr<T> CreateRowwiseOperation( const char* className, IMathEngine& mathEngine )
{
	return dynamic_cast<T*>( CreateRowwiseOperation( className, mathEngine ).Ptr() );
}

//=====================================================================================================================
// Registration mechanisms
// DO NOT use directyle (use macro above)

typedef CPtr<IRowwiseOperation> ( *TCreateRowwiseOperationFunction )( IMathEngine& mathEngine );

void NEOML_API RegisterRowwiseOperation( const char* className, const std::type_info& typeInfo,
	TCreateRowwiseOperationFunction function );

void NEOML_API UnregisterRowwiseOperation( const std::type_info& typeInfo );

template<class T>
class CRowwiseOperationRegistrar {
public:
	explicit CRowwiseOperationRegistrar( const char* className );
	~CRowwiseOperationRegistrar();

private:
	static CPtr<IRowwiseOperation> createObject( IMathEngine& mathEngine ) { return FINE_DEBUG_NEW T( mathEngine ); }
};

template<class T>
inline CRowwiseOperationRegistrar<T>::CRowwiseOperationRegistrar( const char* className )
{
	RegisterRowwiseOperation( className, typeid( T ), createObject );
}

template<class T>
inline CRowwiseOperationRegistrar<T>::~CRowwiseOperationRegistrar()
{
	UnregisterRowwiseOperation( typeid( T ) );
}

} // namespace NeoML
