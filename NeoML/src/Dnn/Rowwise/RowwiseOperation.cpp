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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Rowwise/RowwiseOperation.h>

namespace NeoML {

static CMap<CString, TCreateRowwiseOperationFunction, CDefaultHash<CString>, RuntimeHeap>& getRegisteredRowwise()
{
	static CMap<CString, TCreateRowwiseOperationFunction, CDefaultHash<CString>, RuntimeHeap> registeredRowwise;
	return registeredRowwise;
}

static CMap<const std::type_info*, CString, CDefaultHash<const std::type_info*>, RuntimeHeap>& getRowwiseNames()
{
	static CMap<const std::type_info*, CString, CDefaultHash<const std::type_info*>, RuntimeHeap> rowwiseNames;
	return rowwiseNames;
}

IRowwiseOperation::~IRowwiseOperation() = default;

const char* GetRowwiseOperationName( const IObject* rowwiseOperation )
{
	if( rowwiseOperation == nullptr ) {
		return "";
	}

	const std::type_info& rowwiseType = typeid( *rowwiseOperation );
	TMapPosition pos = getRowwiseNames().GetFirstPosition( &rowwiseType );
	if( pos == NotFound ) {
		return "";
	}
	return getRowwiseNames().GetValue( pos );
}

CPtr<IObject> CreateRowwiseOperation( const char* className, IMathEngine& mathEngine )
{
	TMapPosition pos = getRegisteredRowwise().GetFirstPosition(className);
	if( pos == NotFound ) {
		return 0;
	}
	return getRegisteredRowwise().GetValue( pos )( mathEngine ).Ptr();
}

void RegisterRowwiseOperation( const char* className, const std::type_info& typeInfo,
	TCreateRowwiseOperationFunction function )
{
	NeoAssert( !getRegisteredRowwise().Has( className ) );
	getRegisteredRowwise().Add( className, function );
	getRowwiseNames().Add( &typeInfo, className );
}

void UnregisterRowwiseOperation( const std::type_info& typeInfo )
{
	getRegisteredRowwise().Delete( getRowwiseNames().Get( &typeInfo ) );
	getRowwiseNames().Delete( &typeInfo );
}

} // namespace NeoML
