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

#include <cstring>

// The structure with CPU information
struct CCPUInfo {
	size_t L1CacheSize = 0;
	size_t L2CacheSize = 0;
	size_t L3CacheSize = 0;

	constexpr CCPUInfo() = default;
	constexpr CCPUInfo(size_t L1CacheSize, size_t L2CacheSize = 0, size_t L3CacheSize = 0) :
		L1CacheSize(L1CacheSize),
		L2CacheSize(L2CacheSize),
		L3CacheSize(L3CacheSize)
	{}
};