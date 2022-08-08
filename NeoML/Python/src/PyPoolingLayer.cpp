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

#include "PyPoolingLayer.h"

class CPyPoolingLayer : public CPyLayer {
public:
	explicit CPyPoolingLayer( CPoolingLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLayer( layer, mathEngineOwner ) {}

	int GetFilterHeight() const { return Layer<CPoolingLayer>()->GetFilterHeight(); }
	void SetFilterHeight( int value ) { Layer<CPoolingLayer>()->SetFilterHeight( value ); }
	int GetFilterWidth() const { return Layer<CPoolingLayer>()->GetFilterWidth(); }
	void SetFilterWidth( int value ) { Layer<CPoolingLayer>()->SetFilterWidth( value ); }

	int GetStrideHeight() const { return Layer<CPoolingLayer>()->GetStrideHeight(); }
	void SetStrideHeight( int value ) { Layer<CPoolingLayer>()->SetStrideHeight( value ); }
	int GetStrideWidth() const { return Layer<CPoolingLayer>()->GetStrideWidth(); }
	void SetStrideWidth( int value ) { Layer<CPoolingLayer>()->SetStrideWidth( value ); }
};

//------------------------------------------------------------------------------------------------------------

class CPyMaxPoolingLayer : public CPyPoolingLayer {
public:
	explicit CPyMaxPoolingLayer( CMaxPoolingLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyPoolingLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "MaxPooling" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyMeanPoolingLayer : public CPyPoolingLayer {
public:
	explicit CPyMeanPoolingLayer( CMeanPoolingLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyPoolingLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "MeanPooling" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyGlobalMaxPoolingLayer : public CPyLayer {
public:
	explicit CPyGlobalMaxPoolingLayer(CGlobalMaxPoolingLayer& layer, CPyMathEngineOwner& mathEngineOwner) : CPyLayer(layer, mathEngineOwner) {}

	int GetMaxCount() const { return Layer<CGlobalMaxPoolingLayer>()->GetMaxCount(); }
	void SetMaxCount(int value) { Layer<CGlobalMaxPoolingLayer>()->SetMaxCount(value); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "GlobalMaxPooling" );
		return pyConstructor( py::cast(this), 0 );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyGlobalMeanPoolingLayer : public CPyLayer {
public:
	explicit CPyGlobalMeanPoolingLayer(CGlobalMeanPoolingLayer& layer, CPyMathEngineOwner& mathEngineOwner) : CPyLayer(layer, mathEngineOwner) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "GlobalMeanPooling" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyGlobalSumPoolingLayer : public CPyLayer {
public:
	explicit CPyGlobalSumPoolingLayer(CGlobalSumPoolingLayer& layer, CPyMathEngineOwner& mathEngineOwner) : CPyLayer(layer, mathEngineOwner) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "GlobalSumPooling" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyMaxOverTimePoolingLayer : public CPyLayer {
public:
	explicit CPyMaxOverTimePoolingLayer(CMaxOverTimePoolingLayer& layer, CPyMathEngineOwner& mathEngineOwner) : CPyLayer(layer, mathEngineOwner) {}

	int GetFilterLength() const { return Layer<CMaxOverTimePoolingLayer>()->GetFilterLength(); }
	void SetFilterLength(int value) { Layer<CMaxOverTimePoolingLayer>()->SetFilterLength(value); }
	int GetStrideLength() const { return Layer<CMaxOverTimePoolingLayer>()->GetStrideLength(); }
	void SetStrideLength(int value) { Layer<CMaxOverTimePoolingLayer>()->SetStrideLength(value); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "MaxOverTimePooling" );
		return pyConstructor( py::cast(this), 0, 0 );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyProjectionPoolingLayer : public CPyLayer {
public:
	explicit CPyProjectionPoolingLayer(CProjectionPoolingLayer& layer, CPyMathEngineOwner& mathEngineOwner) : CPyLayer(layer, mathEngineOwner) {}

	int GetDimension() const { return static_cast<int>( Layer<CProjectionPoolingLayer>()->GetDimension() ); }
	void SetDimension(int value) { Layer<CProjectionPoolingLayer>()->SetDimension(static_cast<TBlobDim>(value)); }
	bool GetOriginalSize() const { return Layer<CProjectionPoolingLayer>()->GetRestoreOriginalImageSize(); }
	void SetOriginalSize(bool value) { Layer<CProjectionPoolingLayer>()->SetRestoreOriginalImageSize(value); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ProjectionPooling" );
		return pyConstructor( py::cast(this), 0, 0 );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyPooling3dLayer : public CPyLayer {
public:
	explicit CPyPooling3dLayer( C3dPoolingLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLayer( layer, mathEngineOwner ) {}

	int GetFilterHeight() const { return Layer<C3dPoolingLayer>()->GetFilterHeight(); }
	void SetFilterHeight( int value ) { Layer<C3dPoolingLayer>()->SetFilterHeight( value ); }
	int GetFilterWidth() const { return Layer<C3dPoolingLayer>()->GetFilterWidth(); }
	void SetFilterWidth( int value ) { Layer<C3dPoolingLayer>()->SetFilterWidth( value ); }
	int GetFilterDepth() const { return Layer<C3dPoolingLayer>()->GetFilterDepth(); }
	void SetFilterDepth( int value ) { Layer<C3dPoolingLayer>()->SetFilterDepth( value ); }

	int GetStrideHeight() const { return Layer<C3dPoolingLayer>()->GetStrideHeight(); }
	void SetStrideHeight( int value ) { Layer<C3dPoolingLayer>()->SetStrideHeight( value ); }
	int GetStrideWidth() const { return Layer<C3dPoolingLayer>()->GetStrideWidth(); }
	void SetStrideWidth( int value ) { Layer<C3dPoolingLayer>()->SetStrideWidth( value ); }
	int GetStrideDepth() const { return Layer<C3dPoolingLayer>()->GetStrideDepth(); }
	void SetStrideDepth( int value ) { Layer<C3dPoolingLayer>()->SetStrideDepth( value ); }
};

//------------------------------------------------------------------------------------------------------------

class CPyMaxPooling3dLayer : public CPyPooling3dLayer {
public:
	explicit CPyMaxPooling3dLayer( C3dMaxPoolingLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyPooling3dLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "MaxPooling3D" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyMeanPooling3dLayer : public CPyPooling3dLayer {
public:
	explicit CPyMeanPooling3dLayer( C3dMeanPoolingLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyPooling3dLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "MeanPooling3D" );
		return pyConstructor( py::cast(this) );
	}
};


void InitializePoolingLayer( py::module& m )
{
	py::class_<CPyPoolingLayer, CPyLayer>(m, "Pooling")
		.def( "get_filter_height", &CPyPoolingLayer::GetFilterHeight, py::return_value_policy::reference )
		.def( "get_filter_width", &CPyPoolingLayer::GetFilterWidth, py::return_value_policy::reference )
		.def( "set_filter_height", &CPyPoolingLayer::SetFilterHeight, py::return_value_policy::reference )
		.def( "set_filter_width", &CPyPoolingLayer::SetFilterWidth, py::return_value_policy::reference )

		.def( "get_stride_height", &CPyPoolingLayer::GetStrideHeight, py::return_value_policy::reference )
		.def( "get_stride_width", &CPyPoolingLayer::GetStrideWidth, py::return_value_policy::reference )
		.def( "set_stride_height", &CPyPoolingLayer::SetStrideHeight, py::return_value_policy::reference )
		.def( "set_stride_width", &CPyPoolingLayer::SetStrideWidth, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyMaxPoolingLayer, CPyPoolingLayer>(m, "MaxPooling")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyMaxPoolingLayer( *layer.Layer<CMaxPoolingLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CMaxPoolingLayer> pooling = new CMaxPoolingLayer( mathEngine );
			pooling->SetName( FindFreeLayerName( dnn, "MaxPooling", name ).c_str() );
			pooling->SetFilterHeight( filterHeight );
			pooling->SetFilterWidth( filterWidth );
			pooling->SetStrideHeight( strideHeight );
			pooling->SetStrideWidth( strideWidth );
			dnn.AddLayer( *pooling );

			pooling->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyMaxPoolingLayer( *pooling, layer.MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyMeanPoolingLayer, CPyPoolingLayer>(m, "MeanPooling")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyMeanPoolingLayer( *layer.Layer<CMeanPoolingLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CMeanPoolingLayer> pooling = new CMeanPoolingLayer( mathEngine );
			pooling->SetName( FindFreeLayerName( dnn, "MeanPooling", name ).c_str() );
			pooling->SetFilterHeight( filterHeight );
			pooling->SetFilterWidth( filterWidth );
			pooling->SetStrideHeight( strideHeight );
			pooling->SetStrideWidth( strideWidth );
			dnn.AddLayer( *pooling );

			pooling->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyMeanPoolingLayer( *pooling, layer.MathEngineOwner() );
		}) )
	;

	//------------------------------------------------------------------------------------------------------------

	py::class_<CPyGlobalMaxPoolingLayer, CPyLayer>(m, "GlobalMaxPooling")
		.def(py::init([](const CPyLayer& layer)
		{
			return CPyGlobalMaxPoolingLayer(*layer.Layer<CGlobalMaxPoolingLayer>(), layer.MathEngineOwner());
		}))
		.def(py::init([](const std::string& name, const CPyLayer& layer, int outputNumber, int maxCount)
		{
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CGlobalMaxPoolingLayer> pooling = new CGlobalMaxPoolingLayer(mathEngine);
			pooling->SetName( FindFreeLayerName(dnn, "GlobalMaxPooling", name).c_str() );
			pooling->SetMaxCount(maxCount);
			dnn.AddLayer(*pooling);

			pooling->Connect(0, layer.BaseLayer(), outputNumber);
			return new CPyGlobalMaxPoolingLayer(*pooling, layer.MathEngineOwner());
		}))
		.def("get_max_count", &CPyGlobalMaxPoolingLayer::GetMaxCount, py::return_value_policy::reference)
		.def("set_max_count", &CPyGlobalMaxPoolingLayer::SetMaxCount, py::return_value_policy::reference)
	;

	//------------------------------------------------------------------------------------------------------------

	py::class_<CPyGlobalMeanPoolingLayer, CPyLayer>(m, "GlobalMeanPooling")
		.def(py::init([](const CPyLayer& layer)
		{
			return CPyGlobalMeanPoolingLayer(*layer.Layer<CGlobalMeanPoolingLayer>(), layer.MathEngineOwner());
		}))
		.def(py::init([](const std::string& name, const CPyLayer& layer, int outputNumber)
		{
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CGlobalMeanPoolingLayer> pooling = new CGlobalMeanPoolingLayer(mathEngine);
			pooling->SetName( FindFreeLayerName(dnn, "GlobalMeanPooling", name).c_str() );
			dnn.AddLayer(*pooling);

			pooling->Connect(0, layer.BaseLayer(), outputNumber);
			return new CPyGlobalMeanPoolingLayer(*pooling, layer.MathEngineOwner());
		}))
	;

	//------------------------------------------------------------------------------------------------------------

	py::class_<CPyGlobalSumPoolingLayer, CPyLayer>(m, "GlobalSumPooling")
		.def(py::init([](const CPyLayer& layer)
		{
			return CPyGlobalSumPoolingLayer(*layer.Layer<CGlobalSumPoolingLayer>(), layer.MathEngineOwner());
		}))
		.def(py::init([](const std::string& name, const CPyLayer& layer, int outputNumber)
		{
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CGlobalSumPoolingLayer> pooling = new CGlobalSumPoolingLayer(mathEngine);
			pooling->SetName( FindFreeLayerName(dnn, "GlobalSumPooling", name).c_str() );
			dnn.AddLayer(*pooling);

			pooling->Connect(0, layer.BaseLayer(), outputNumber);
			return new CPyGlobalSumPoolingLayer(*pooling, layer.MathEngineOwner());
		}))
	;

	//------------------------------------------------------------------------------------------------------------

	py::class_<CPyMaxOverTimePoolingLayer, CPyLayer>(m, "MaxOverTimePooling")
		.def(py::init([](const CPyLayer& layer)
		{
			return CPyMaxOverTimePoolingLayer(*layer.Layer<CMaxOverTimePoolingLayer>(), layer.MathEngineOwner());
		}))
		.def(py::init([](const std::string& name, const CPyLayer& layer, int outputNumber, int filter_len, int stride_len)
		{
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CMaxOverTimePoolingLayer> pooling = new CMaxOverTimePoolingLayer(mathEngine);
			pooling->SetName( FindFreeLayerName(dnn, "MaxOverTimePooling", name).c_str() );
			pooling->SetFilterLength(filter_len);
			pooling->SetStrideLength(stride_len);
			dnn.AddLayer(*pooling);

			pooling->Connect(0, layer.BaseLayer(), outputNumber);
			return new CPyMaxOverTimePoolingLayer(*pooling, layer.MathEngineOwner());
		}))
		.def("get_filter_len", &CPyMaxOverTimePoolingLayer::GetFilterLength, py::return_value_policy::reference)
		.def("set_filter_len", &CPyMaxOverTimePoolingLayer::SetFilterLength, py::return_value_policy::reference)
		.def("get_stride_len", &CPyMaxOverTimePoolingLayer::GetStrideLength, py::return_value_policy::reference)
		.def("set_stride_len", &CPyMaxOverTimePoolingLayer::SetStrideLength, py::return_value_policy::reference)
	;

	//------------------------------------------------------------------------------------------------------------

	py::class_<CPyProjectionPoolingLayer, CPyLayer>(m, "ProjectionPooling")
		.def(py::init([](const CPyLayer& layer)
		{
			return CPyProjectionPoolingLayer(*layer.Layer<CProjectionPoolingLayer>(), layer.MathEngineOwner());
		}))
		.def(py::init([](const std::string& name, const CPyLayer& layer, int outputNumber, int dimension, bool originalSize)
		{
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CProjectionPoolingLayer> pooling = new CProjectionPoolingLayer(mathEngine);
			pooling->SetName( FindFreeLayerName(dnn, "ProjectionPooling", name).c_str() );
			pooling->SetDimension(static_cast<TBlobDim>(dimension));
			pooling->SetRestoreOriginalImageSize(originalSize);
			dnn.AddLayer(*pooling);

			pooling->Connect(0, layer.BaseLayer(), outputNumber);
			return new CPyProjectionPoolingLayer(*pooling, layer.MathEngineOwner());
		}))
		.def("get_dimension", &CPyProjectionPoolingLayer::GetDimension, py::return_value_policy::reference)
		.def("set_dimension", &CPyProjectionPoolingLayer::SetDimension, py::return_value_policy::reference)
		.def("get_original_size", &CPyProjectionPoolingLayer::GetOriginalSize, py::return_value_policy::reference)
		.def("set_original_size", &CPyProjectionPoolingLayer::SetOriginalSize, py::return_value_policy::reference)
	;

	//------------------------------------------------------------------------------------------------------------

	py::class_<CPyPooling3dLayer, CPyLayer>(m, "Pooling3D")
		.def("get_filter_height", &CPyPooling3dLayer::GetFilterHeight, py::return_value_policy::reference)
		.def("get_filter_width", &CPyPooling3dLayer::GetFilterWidth, py::return_value_policy::reference)
		.def("get_filter_depth", &CPyPooling3dLayer::GetFilterDepth, py::return_value_policy::reference)
		.def("set_filter_height", &CPyPooling3dLayer::SetFilterHeight, py::return_value_policy::reference)
		.def("set_filter_width", &CPyPooling3dLayer::SetFilterWidth, py::return_value_policy::reference)
		.def("set_filter_depth", &CPyPooling3dLayer::SetFilterDepth, py::return_value_policy::reference)

		.def("get_stride_height", &CPyPooling3dLayer::GetStrideHeight, py::return_value_policy::reference)
		.def("get_stride_width", &CPyPooling3dLayer::GetStrideWidth, py::return_value_policy::reference)
		.def("get_stride_depth", &CPyPooling3dLayer::GetStrideDepth, py::return_value_policy::reference)
		.def("set_stride_height", &CPyPooling3dLayer::SetStrideHeight, py::return_value_policy::reference)
		.def("set_stride_width", &CPyPooling3dLayer::SetStrideWidth, py::return_value_policy::reference)
		.def("set_stride_depth", &CPyPooling3dLayer::SetStrideDepth, py::return_value_policy::reference)
	;

	//------------------------------------------------------------------------------------------------------------

	py::class_<CPyMaxPooling3dLayer, CPyPooling3dLayer>(m, "MaxPooling3D")
		.def(py::init([](const CPyLayer& layer)
		{
			return CPyMaxPooling3dLayer(*layer.Layer<C3dMaxPoolingLayer>(), layer.MathEngineOwner());
		}))
		.def(py::init([](const std::string& name, const CPyLayer& layer, int outputNumber, int filterHeight, int filterWidth, int filterDepth,
			int strideHeight, int strideWidth, int strideDepth)
		{
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<C3dMaxPoolingLayer> pooling = new C3dMaxPoolingLayer(mathEngine);
			pooling->SetFilterHeight(filterHeight);
			pooling->SetFilterWidth(filterWidth);
			pooling->SetFilterDepth(filterDepth);
			pooling->SetStrideHeight(strideHeight);
			pooling->SetStrideWidth(strideWidth);
			pooling->SetStrideDepth(strideDepth);
			pooling->SetName( FindFreeLayerName(dnn, "MaxPooling3D", name).c_str() );
			dnn.AddLayer(*pooling);

			pooling->Connect(0, layer.BaseLayer(), outputNumber);
			return new CPyMaxPooling3dLayer(*pooling, layer.MathEngineOwner());
		}))
	;

	//------------------------------------------------------------------------------------------------------------

	py::class_<CPyMeanPooling3dLayer, CPyPooling3dLayer>(m, "MeanPooling3D")
		.def(py::init([](const CPyLayer& layer)
		{
			return CPyMeanPooling3dLayer(*layer.Layer<C3dMeanPoolingLayer>(), layer.MathEngineOwner());
		}))
		.def(py::init([](const std::string& name, const CPyLayer& layer, int outputNumber, int filterHeight, int filterWidth, int filterDepth,
			int strideHeight, int strideWidth, int strideDepth)
		{
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<C3dMeanPoolingLayer> pooling = new C3dMeanPoolingLayer(mathEngine);
			pooling->SetFilterHeight(filterHeight);
			pooling->SetFilterWidth(filterWidth);
			pooling->SetFilterDepth(filterDepth);
			pooling->SetStrideHeight(strideHeight);
			pooling->SetStrideWidth(strideWidth);
			pooling->SetStrideDepth(strideDepth);
			pooling->SetName( FindFreeLayerName(dnn, "MeanPooling3D", name).c_str() );
			dnn.AddLayer(*pooling);

			pooling->Connect(0, layer.BaseLayer(), outputNumber);
			return new CPyMeanPooling3dLayer(*pooling, layer.MathEngineOwner());
		}))
	;
}
