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

namespace NeoML {

// MathEngine blob data types
enum TBlobType {
	CT_Invalid = 0,
	CT_Float,
	CT_Int,
};

// Data types used in MathEngine
template<class T>
struct CBlobType {
	// typedef for the base data type used in Math Engine
	typedef T TDataType;

	// Gets the blob data type
	static TBlobType GetType() { return CT_Invalid; }
};

// The float data type description
template<>
struct CBlobType<float> {
	// typedef for the base data type used in Math Engine
	typedef float TDataType;

	// Gets the blob data type
	static TBlobType GetType() { return CT_Float; }
};

template<>
struct CBlobType<const float> {
	// typedef for the base data type used in Math Engine
	typedef float TDataType;

	// Gets the blob data type
	static TBlobType GetType() { return CT_Float; }
};

// The int data type description
template<>
struct CBlobType<int> {
	// typedef for the base data type used in Math Engine
	typedef int TDataType;

	// Gets the blob data type
	static TBlobType GetType() { return CT_Int; }
};

template<>
struct CBlobType<const int> {
	// typedef for the base data type used in Math Engine
	typedef int TDataType;

	// Gets the blob data type
	static TBlobType GetType() { return CT_Int; }
};

} // namespace NeoML
