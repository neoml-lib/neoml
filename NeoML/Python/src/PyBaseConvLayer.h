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

#include "PyLayer.h"

class CPyBaseConvLayer : public CPyLayer {
public:
	explicit CPyBaseConvLayer( CBaseConvLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	int GetFilterHeight() const { return Layer<CBaseConvLayer>()->GetFilterHeight(); }
	void SetFilterHeight( int value ) { Layer<CBaseConvLayer>()->SetFilterHeight( value ); }
	int GetFilterWidth() const { return Layer<CBaseConvLayer>()->GetFilterWidth(); }
	void SetFilterWidth( int value ) { Layer<CBaseConvLayer>()->SetFilterWidth( value ); }

	int GetStrideHeight() const { return Layer<CBaseConvLayer>()->GetStrideHeight(); }
	void SetStrideHeight( int value ) { Layer<CBaseConvLayer>()->SetStrideHeight( value ); }
	int GetStrideWidth() const { return Layer<CBaseConvLayer>()->GetStrideWidth(); }
	void SetStrideWidth( int value ) { Layer<CBaseConvLayer>()->SetStrideWidth( value ); }

	int GetPaddingHeight() const { return Layer<CBaseConvLayer>()->GetPaddingHeight(); }
	void SetPaddingHeight( int value ) { Layer<CBaseConvLayer>()->SetPaddingHeight( value ); }
	int GetPaddingWidth() const { return Layer<CBaseConvLayer>()->GetPaddingWidth(); }
	void SetPaddingWidth( int value ) { Layer<CBaseConvLayer>()->SetPaddingWidth( value ); }

	int GetDilationHeight() const { return Layer<CBaseConvLayer>()->GetDilationHeight(); }
	void SetDilationHeight( int value ) { Layer<CBaseConvLayer>()->SetDilationHeight( value ); }
	int GetDilationWidth() const { return Layer<CBaseConvLayer>()->GetDilationWidth(); }
	void SetDilationWidth( int value ) { Layer<CBaseConvLayer>()->SetDilationWidth( value ); }

	int GetFilterCount() const { return Layer<CBaseConvLayer>()->GetFilterCount(); }
	void SetFilterCount( int value ) { Layer<CBaseConvLayer>()->SetFilterCount( value ); }

	py::object GetFilter() const;
	py::object GetFreeTerm() const;
};

//------------------------------------------------------------------------------------------------------------

void InitializeBaseConvLayer( py::module& m );