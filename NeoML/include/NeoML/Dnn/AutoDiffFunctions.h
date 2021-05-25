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

// Creates a const blob filled with the specified value.
NEOML_API CPtr<const CDnnBlob> Const( IMathEngine& mathEngine, float data, const CBlobDesc& desc = {1} );
// Creates a const blob using the specified data.
NEOML_API CPtr<const CDnnBlob> Const( IMathEngine& mathEngine, const float* data, const CBlobDesc& desc );

// Creates a blob which is the elementwise sum of the specified blobs. 
// Input blob sizes should be the same.
// res[i] = first[i] + second[i]
NEOML_API CPtr<const CDnnBlob> Add( const CDnnBlob* first, const CDnnBlob* second );
// Creates a blob that is the elementwise sum of the input blob with the given scalar value.
// res[i] = first[i] + second
NEOML_API CPtr<const CDnnBlob> Add( const CDnnBlob* first, float second );
// res[i] = first + second[i]
NEOML_API CPtr<const CDnnBlob> Add( float first, const CDnnBlob* second );

// Creates a blob which is the elementwise subtraction of the specified blobs. 
// Input blob sizes should be the same.
// res[i] = first[i] - second[i]
NEOML_API CPtr<const CDnnBlob> Sub( const CDnnBlob* first, const CDnnBlob* second );
// Creates a blob that is the elementwise subtraction of the input blob and the given scalar value.
// res[i] = first[i] - second
NEOML_API CPtr<const CDnnBlob> Sub( const CDnnBlob* first, float second );
// res[i] = first - second[i]
NEOML_API CPtr<const CDnnBlob> Sub( float first, const CDnnBlob* second );

// Creates a blob which is the elementwise product of the specified blobs. 
// Input blob sizes should be the same.
// res[i] = first[i] * second[i]
CPtr<const CDnnBlob> NEOML_API Mul( const CDnnBlob* first, const CDnnBlob* second );
// Creates a blob that is the elementwise product of the input blob and the given scalar value.
// res[i] = first[i] * second
CPtr<const CDnnBlob> NEOML_API Mul( const CDnnBlob* first, float second );
// res[i] = first * second[i]
CPtr<const CDnnBlob> NEOML_API Mul( float first, const CDnnBlob* second );

// Creates the blob which is the result of elementwise division of the first blob by the second. 
// Input blob sizes should be the same.
// res[i] = first[i] / second[i]
CPtr<const CDnnBlob> NEOML_API Div( const CDnnBlob* first, const CDnnBlob* second );
// Creates the blob which is the result of elementwise division of the input blob by the given scalar value, or the other way round.
// res[i] = first[i] / second
CPtr<const CDnnBlob> NEOML_API Div( const CDnnBlob* first, float second );
// res[i] = first / second[i]
CPtr<const CDnnBlob> NEOML_API Div( float first, const CDnnBlob* second );

// Creates the blob which is the elementwise maximum of the specified inputs. 
// res[i] = max(first[i], second)
CPtr<const CDnnBlob> NEOML_API Max( const CDnnBlob* first, float second );
// res[i] = max(first, second[i])
CPtr<const CDnnBlob> NEOML_API Max( float first, const CDnnBlob* second );

// Calculates the total sum of all blob elements and returns it as a single-element blob.
NEOML_API CPtr<const CDnnBlob> Sum( const CDnnBlob* first );

// Creates the blob each element of which is the negative value of the corresponding element of the specified blob.
// res[i] = -first[i]
NEOML_API CPtr<const CDnnBlob> Neg( const CDnnBlob* first );

// Creates the blob each element of which is the absolute value of the corresponding element of the specified blob.
// res[i] = |first[i]|
NEOML_API CPtr<const CDnnBlob> Abs( const CDnnBlob* first );

// Creates the blob each element of which is the log of the corresponding element of the specified blob.
// res[i] = log(first[i])
NEOML_API CPtr<const CDnnBlob> Log( const CDnnBlob* first );

// Creates the blob each element of which is the exponential of the corresponding element of the specified blob.
// res[i] = exp(first[i])
NEOML_API CPtr<const CDnnBlob> Exp( const CDnnBlob* first );

// Finds values of the k largest elements in the blob.
// The result is a blob of size k.
CPtr<const CDnnBlob> NEOML_API TopK( const CDnnBlob* first, int k );

// Creates the blob each element of which is the clipped value of the corresponding element of the specified blob.
// res[i] = min( max(first[i], minValue), maxValue )
NEOML_API CPtr<const CDnnBlob> Clip( const CDnnBlob* first, float minValue, float maxValue );

// Calculates the binary cross-entropy for two blobs: the first one contains correct labels, the second one contains predictions.
// result = (1 - labels) * x + log(1 + exp(-x))
// if fromLogits then x = preds else x = log( clippedPreds / (1 - clippedPreds) )
// Blobs should be of the same shape.
NEOML_API CPtr<const CDnnBlob> BinaryCrossEntropy( const CDnnBlob* labels, const CDnnBlob* preds, bool fromLogits );

} // namespace NeoML
