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

#include <common.h>
#pragma hdrstop

#include "PyBaseConvLayer.h"
#include "PyDnnBlob.h"

void CPyBaseConvLayer::SetFilter( CPyBlob& value ) const
{
	Layer<CBaseConvLayer>()->SetFilterData( value.Blob() );
}

void CPyBaseConvLayer::SetFreeTerm( CPyBlob& value ) const
{
	Layer<CBaseConvLayer>()->SetFreeTermData( value.Blob() );
}

void CPyBaseConvLayer::ApplyBatchNormalization(const CPyLayer& layer)
{
	Layer<CBaseConvLayer>()->ApplyBatchNormalization(*layer.Layer<CBatchNormalizationLayer>());
}

//------------------------------------------------------------------------------------------------------------

void InitializeBaseConvLayer( py::module& m )
{
	py::class_<CPyBaseConvLayer, CPyLayer>(m, "BaseConv")
		.def( "get_filter_count", &CPyBaseConvLayer::GetFilterCount, py::return_value_policy::reference )
		.def( "set_filter_count", &CPyBaseConvLayer::SetFilterCount, py::return_value_policy::reference )

		.def( "get_filter_height", &CPyBaseConvLayer::GetFilterHeight, py::return_value_policy::reference )
		.def( "get_filter_width", &CPyBaseConvLayer::GetFilterWidth, py::return_value_policy::reference )
		.def( "set_filter_height", &CPyBaseConvLayer::SetFilterHeight, py::return_value_policy::reference )
		.def( "set_filter_width", &CPyBaseConvLayer::SetFilterWidth, py::return_value_policy::reference )

		.def( "get_stride_height", &CPyBaseConvLayer::GetStrideHeight, py::return_value_policy::reference )
		.def( "get_stride_width", &CPyBaseConvLayer::GetStrideWidth, py::return_value_policy::reference )
		.def( "set_stride_height", &CPyBaseConvLayer::SetStrideHeight, py::return_value_policy::reference )
		.def( "set_stride_width", &CPyBaseConvLayer::SetStrideWidth, py::return_value_policy::reference )

		.def( "get_padding_height", &CPyBaseConvLayer::GetPaddingHeight, py::return_value_policy::reference )
		.def( "get_padding_width", &CPyBaseConvLayer::GetPaddingWidth, py::return_value_policy::reference )
		.def( "set_padding_height", &CPyBaseConvLayer::SetPaddingHeight, py::return_value_policy::reference )
		.def( "set_padding_width", &CPyBaseConvLayer::SetPaddingWidth, py::return_value_policy::reference )

		.def( "get_dilation_height", &CPyBaseConvLayer::GetDilationHeight, py::return_value_policy::reference )
		.def( "get_dilation_width", &CPyBaseConvLayer::GetDilationWidth, py::return_value_policy::reference )
		.def( "set_dilation_height", &CPyBaseConvLayer::SetDilationHeight, py::return_value_policy::reference )
		.def( "set_dilation_width", &CPyBaseConvLayer::SetDilationWidth, py::return_value_policy::reference )

		.def( "set_filter", &CPyBaseConvLayer::SetFilter, py::return_value_policy::reference )
		.def( "get_filter", &CPyBaseConvLayer::GetFilter, py::return_value_policy::reference )
		.def( "set_free_term", &CPyBaseConvLayer::SetFreeTerm, py::return_value_policy::reference )
		.def( "get_free_term", &CPyBaseConvLayer::GetFreeTerm, py::return_value_policy::reference )
		.def( "set_filter", &CPyBaseConvLayer::SetFilter, py::return_value_policy::reference )
		.def( "set_free_term", &CPyBaseConvLayer::SetFreeTerm, py::return_value_policy::reference )

		.def( "apply_batch_normalization", &CPyBaseConvLayer::ApplyBatchNormalization, py::return_value_policy::reference )
	;
}
