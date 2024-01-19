/* Copyright © 2024 ABBYY

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

#include <initializer_list>
#include <NeoML/NeoML.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

template <typename T>
class CLayerWrapper;

template <typename... Ts>
class CDnnHead final {
public:
	CDnnHead( CLayerWrapper<Ts>... linearWrappers )
	{
		CBaseLayer* inputLayer = nullptr;
		CBaseLayer* layers[]{ linearWrappers( inputLayer )... };
		( void ) layers;
		// TODO: ???
	}

	CBaseLayer* operator()( std::initializer_list<CBaseLayer*> inputs )
	{
		for( CBaseLayer* input : inputs ) {
			( void ) input;
			// TODO: ???
		}
		return nullptr;
	}
};

} // namespace NeoML

