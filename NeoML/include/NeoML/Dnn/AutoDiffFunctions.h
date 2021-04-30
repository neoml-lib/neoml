/* Copyright © 2017-2021 ABBYY Production LLC

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
#include <NeoML/Dnn/DnnBlob.h>

namespace NeoML {

NEOML_API CPtr<const CDnnBlob> Const( IMathEngine& mathEngine, float data, const CBlobDesc& desc = {1} );
NEOML_API CPtr<const CDnnBlob> Const( IMathEngine& mathEngine, float* data, const CBlobDesc& desc );
NEOML_API CPtr<const CDnnBlob> Const( IMathEngine& mathEngine, const CArray<float>& data, const CBlobDesc& desc );

NEOML_API CPtr<const CDnnBlob> Add( const CDnnBlob* first, const CDnnBlob* second );
NEOML_API CPtr<const CDnnBlob> Add( const CDnnBlob* first, float second );
NEOML_API CPtr<const CDnnBlob> Add( float first, const CDnnBlob* second );

NEOML_API CPtr<const CDnnBlob> Sub( const CDnnBlob* first, const CDnnBlob* second );
NEOML_API CPtr<const CDnnBlob> Sub( const CDnnBlob* first, float second );
NEOML_API CPtr<const CDnnBlob> Sub( float first, const CDnnBlob* second );

CPtr<const CDnnBlob> NEOML_API Mult( const CDnnBlob* first, const CDnnBlob* second );
CPtr<const CDnnBlob> NEOML_API Mult( const CDnnBlob* first, float second );
CPtr<const CDnnBlob> NEOML_API Mult( float first, const CDnnBlob* second );

CPtr<const CDnnBlob> NEOML_API Div( const CDnnBlob* first, const CDnnBlob* second );
CPtr<const CDnnBlob> NEOML_API Div( const CDnnBlob* first, float second );
CPtr<const CDnnBlob> NEOML_API Div( float first, const CDnnBlob* second );

CPtr<const CDnnBlob> NEOML_API Max( const CDnnBlob* first, float second );
CPtr<const CDnnBlob> NEOML_API Max( float first, const CDnnBlob* second );

NEOML_API CPtr<const CDnnBlob> Sum( const CDnnBlob* first );

NEOML_API CPtr<const CDnnBlob> Neg( const CDnnBlob* first );

NEOML_API CPtr<const CDnnBlob> Abs( const CDnnBlob* first );

NEOML_API CPtr<const CDnnBlob> Log( const CDnnBlob* first );

NEOML_API CPtr<const CDnnBlob> Exp( const CDnnBlob* first );

CPtr<const CDnnBlob> NEOML_API TopK( const CDnnBlob* first, int k );

NEOML_API CPtr<const CDnnBlob> Clip( const CDnnBlob* first, float min, float max );

NEOML_API CPtr<const CDnnBlob> BinaryCrossEntropy( const CDnnBlob* labels, const CDnnBlob* preds, bool fromLogits );

} // namespace NeoML
