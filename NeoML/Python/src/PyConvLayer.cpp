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

#include "PyConvLayer.h"

class CPyConvLayer : public CPyBaseConvLayer {
public:
	explicit CPyConvLayer( CConvLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyBaseConvLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Conv" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyBaseConv3DLayer : public CPyBaseConvLayer {
public:
	explicit CPyBaseConv3DLayer( CBase3dConvLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyBaseConvLayer( layer, mathEngineOwner ) {}

	int GetFilterDepth() const { return Layer<CBase3dConvLayer>()->GetFilterDepth(); }
	void SetFilterDepth( int value ) { Layer<CBase3dConvLayer>()->SetFilterDepth( value ); }

	int GetStrideDepth() const { return Layer<CBase3dConvLayer>()->GetStrideDepth(); }
	void SetStrideDepth( int value ) { Layer<CBase3dConvLayer>()->SetStrideDepth( value ); }

	int GetPaddingDepth() const { return Layer<CBase3dConvLayer>()->GetPaddingDepth(); }
	void SetPaddingDepth( int value ) { Layer<CBase3dConvLayer>()->SetPaddingDepth( value ); }
};

class CPyConv3DLayer : public CPyBaseConv3DLayer {
public:
	explicit CPyConv3DLayer( C3dConvLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyBaseConv3DLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Conv3D" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyTransposedConv3DLayer : public CPyBaseConv3DLayer {
public:
	explicit CPyTransposedConv3DLayer( C3dTransposedConvLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyBaseConv3DLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "TransposedConv3D" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyTransposedConvLayer : public CPyBaseConvLayer {
public:
	explicit CPyTransposedConvLayer( CTransposedConvLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyBaseConvLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "TransposedConv" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyTimeConvLayer : public CPyLayer {
public:
	explicit CPyTimeConvLayer( CTimeConvLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	int GetFilterCount() const { return Layer<CTimeConvLayer>()->GetFilterCount(); }
	void SetFilterCount( int value ) { Layer<CTimeConvLayer>()->SetFilterCount( value ); }

	int GetFilterSize() const { return Layer<CTimeConvLayer>()->GetFilterSize(); }
	void SetFilterSize( int value ) { Layer<CTimeConvLayer>()->SetFilterSize( value ); }

	int GetStride() const { return Layer<CTimeConvLayer>()->GetStride(); }
	void SetStride( int value ) { Layer<CTimeConvLayer>()->SetStride( value ); }

	void SetPaddingFront( int value ) { Layer<CTimeConvLayer>()->SetPaddingFront( value ); }
	int GetPaddingFront() const { return Layer<CTimeConvLayer>()->GetPaddingFront(); }

	void SetPaddingBack( int value ) { Layer<CTimeConvLayer>()->SetPaddingBack( value ); }
	int GetPaddingBack() const { return Layer<CTimeConvLayer>()->GetPaddingBack(); }

	int GetDilation() const { return Layer<CTimeConvLayer>()->GetDilation(); }
	void SetDilation( int value ) { Layer<CTimeConvLayer>()->SetDilation( value ); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "TimeConv" );
		return pyConstructor( py::cast(this), 0, 0 );
	}
};

class CPyChannelwiseConvLayer : public CPyBaseConvLayer {
public:
	explicit CPyChannelwiseConvLayer( CChannelwiseConvLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyBaseConvLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ChannelwiseConv" );
		return pyConstructor( py::cast(this), 0 );
	}
};

void InitializeConvLayer( py::module& m )
{
	py::class_<CPyConvLayer, CPyBaseConvLayer>(m, "Conv")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyConvLayer( *layer.Layer<CConvLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, int filterCount, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth, int paddingHeight, int paddingWidth, int dilationHeight, int dilationWidth, bool freeTerm )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CConvLayer> conv = new CConvLayer( mathEngine );
			conv->SetName( FindFreeLayerName( dnn, "Conv", name ).c_str() );
			conv->SetFilterCount( filterCount );
			conv->SetFilterHeight( filterHeight );
			conv->SetFilterWidth( filterWidth );
			conv->SetStrideHeight( strideHeight );
			conv->SetStrideWidth( strideWidth );
			conv->SetPaddingHeight( paddingHeight );
			conv->SetPaddingWidth( paddingWidth );
			conv->SetDilationHeight( dilationHeight );
			conv->SetDilationWidth( dilationWidth );
			conv->SetZeroFreeTerm( freeTerm );

			dnn.AddLayer( *conv );
			for( int i = 0; i < layers.size(); i++ ) {
				conv->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}
			return CPyConvLayer( *conv, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

	py::class_<CPyBaseConv3DLayer, CPyBaseConvLayer>(m, "BaseConv3D")
		.def("get_filter_depth", &CPyBaseConv3DLayer::GetFilterDepth, py::return_value_policy::reference)
		.def("set_filter_depth", &CPyBaseConv3DLayer::SetFilterDepth, py::return_value_policy::reference)
		.def("get_stride_depth", &CPyBaseConv3DLayer::GetStrideDepth, py::return_value_policy::reference)
		.def("set_stride_depth", &CPyBaseConv3DLayer::SetStrideDepth, py::return_value_policy::reference)
		.def("get_padding_depth", &CPyBaseConv3DLayer::GetPaddingDepth, py::return_value_policy::reference)
		.def("set_padding_depth", &CPyBaseConv3DLayer::SetPaddingDepth, py::return_value_policy::reference)
	;

	py::class_<CPyConv3DLayer, CPyBaseConv3DLayer>(m, "Conv3D")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyConv3DLayer( *layer.Layer<C3dConvLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, int filterCount,
			int filterHeight, int filterWidth, int filterDepth,
			int strideHeight, int strideWidth, int strideDepth, int paddingHeight, int paddingWidth, int paddingDepth,
			bool freeTerm )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<C3dConvLayer> conv = new C3dConvLayer( mathEngine );
			conv->SetName( FindFreeLayerName( dnn, "Conv3D", name ).c_str() );
			conv->SetFilterCount( filterCount );
			conv->SetFilterHeight( filterHeight );
			conv->SetFilterWidth( filterWidth );
			conv->SetFilterDepth( filterDepth );
			conv->SetStrideHeight( strideHeight );
			conv->SetStrideWidth( strideWidth );
			conv->SetStrideDepth( strideDepth );
			conv->SetPaddingHeight( paddingHeight );
			conv->SetPaddingWidth( paddingWidth );
			conv->SetPaddingDepth( paddingDepth );
			conv->SetZeroFreeTerm( freeTerm );

			dnn.AddLayer( *conv );
			for( int i = 0; i < layers.size(); i++ ) {
				conv->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}
			return CPyConv3DLayer( *conv, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}))
	;

	py::class_<CPyTransposedConv3DLayer, CPyBaseConv3DLayer>(m, "TransposedConv3D")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyTransposedConv3DLayer( *layer.Layer<C3dTransposedConvLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, int filterCount,
			int filterHeight, int filterWidth, int filterDepth,
			int strideHeight, int strideWidth, int strideDepth, int paddingHeight, int paddingWidth, int paddingDepth,
			bool freeTerm )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<C3dTransposedConvLayer> conv = new C3dTransposedConvLayer( mathEngine );
			conv->SetName( FindFreeLayerName( dnn, "TransposedConv3D", name ).c_str() );
			conv->SetFilterCount( filterCount );
			conv->SetFilterHeight( filterHeight );
			conv->SetFilterWidth( filterWidth );
			conv->SetFilterDepth( filterDepth );
			conv->SetStrideHeight( strideHeight );
			conv->SetStrideWidth( strideWidth );
			conv->SetStrideDepth( strideDepth );
			conv->SetPaddingHeight( paddingHeight );
			conv->SetPaddingWidth( paddingWidth );
			conv->SetPaddingDepth( paddingDepth );
			conv->SetZeroFreeTerm( freeTerm );

			dnn.AddLayer( *conv );
			for( int i = 0; i < layers.size(); i++ ) {
				conv->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}
			return CPyTransposedConv3DLayer( *conv, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}))
	;

	py::class_<CPyTransposedConvLayer, CPyBaseConvLayer>(m, "TransposedConv")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyTransposedConvLayer( *layer.Layer<CTransposedConvLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, int filterCount, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth, int paddingHeight, int paddingWidth, int dilationHeight, int dilationWidth, bool freeTerm )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CTransposedConvLayer> conv = new CTransposedConvLayer( mathEngine );
			conv->SetName( FindFreeLayerName( dnn, "TransposedConv", name ).c_str() );
			conv->SetFilterCount( filterCount );
			conv->SetFilterHeight( filterHeight );
			conv->SetFilterWidth( filterWidth );
			conv->SetStrideHeight( strideHeight );
			conv->SetStrideWidth( strideWidth );
			conv->SetPaddingHeight( paddingHeight );
			conv->SetPaddingWidth( paddingWidth );
			conv->SetDilationHeight( dilationHeight );
			conv->SetDilationWidth( dilationWidth );
			conv->SetZeroFreeTerm( freeTerm );

			dnn.AddLayer( *conv );
			for( int i = 0; i < layers.size(); i++ ) {
				conv->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}
			return CPyTransposedConvLayer( *conv, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

	py::class_<CPyTimeConvLayer, CPyLayer>(m, "TimeConv")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyTimeConvLayer( *layer.Layer<CTimeConvLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, int filterCount, int filterSize,
			int paddingFront, int paddingBack, int stride, int dilation )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CTimeConvLayer> conv = new CTimeConvLayer( mathEngine );
			conv->SetName( FindFreeLayerName( dnn, "TimeConv", name ).c_str() );
			conv->SetFilterCount( filterCount );
			conv->SetFilterSize( filterSize );
			conv->SetStride( stride );
			conv->SetPaddingFront( paddingFront );
			conv->SetPaddingBack( paddingBack );
			conv->SetDilation( dilation );

			dnn.AddLayer( *conv );
			for( int i = 0; i < layers.size(); i++ ) {
				conv->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}
			return CPyTimeConvLayer( *conv, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def("get_filter_count", &CPyTimeConvLayer::GetFilterCount, py::return_value_policy::reference)
		.def("set_filter_count", &CPyTimeConvLayer::SetFilterCount, py::return_value_policy::reference)
		.def("get_filter_size", &CPyTimeConvLayer::GetFilterSize, py::return_value_policy::reference)
		.def("set_filter_size", &CPyTimeConvLayer::SetFilterSize, py::return_value_policy::reference)
		.def("get_stride", &CPyTimeConvLayer::GetStride, py::return_value_policy::reference)
		.def("set_stride", &CPyTimeConvLayer::SetStride, py::return_value_policy::reference)
		.def("get_padding_front", &CPyTimeConvLayer::GetPaddingFront, py::return_value_policy::reference)
		.def("set_padding_front", &CPyTimeConvLayer::SetPaddingFront, py::return_value_policy::reference)
		.def("get_padding_back", &CPyTimeConvLayer::GetPaddingBack, py::return_value_policy::reference)
		.def("set_padding_back", &CPyTimeConvLayer::SetPaddingBack, py::return_value_policy::reference)
		.def("get_dilation", &CPyTimeConvLayer::GetDilation, py::return_value_policy::reference)
		.def("set_dilation", &CPyTimeConvLayer::SetDilation, py::return_value_policy::reference)
	;

	py::class_<CPyChannelwiseConvLayer, CPyBaseConvLayer>(m, "ChannelwiseConv")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyChannelwiseConvLayer( *layer.Layer<CChannelwiseConvLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, int filterCount, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth, int paddingHeight, int paddingWidth, bool freeTerm )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CChannelwiseConvLayer> conv = new CChannelwiseConvLayer( mathEngine );
			conv->SetName( FindFreeLayerName( dnn, "ChannelwiseConv", name ).c_str() );
			conv->SetFilterCount( filterCount );
			conv->SetFilterHeight( filterHeight );
			conv->SetFilterWidth( filterWidth );
			conv->SetStrideHeight( strideHeight );
			conv->SetStrideWidth( strideWidth );
			conv->SetPaddingHeight( paddingHeight );
			conv->SetPaddingWidth( paddingWidth );
			conv->SetZeroFreeTerm( freeTerm );

			dnn.AddLayer( *conv );
			for( int i = 0; i < layers.size(); i++ ) {
				conv->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}
			return CPyChannelwiseConvLayer( *conv, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

}
