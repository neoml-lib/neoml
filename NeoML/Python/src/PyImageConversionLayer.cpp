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

#include "PyImageConversionLayer.h"

class CPyImageToPixelLayer : public CPyLayer {
public:
	explicit CPyImageToPixelLayer( CImageToPixelLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ImageToPixel" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyPixelToImageLayer : public CPyLayer {
public:
	explicit CPyPixelToImageLayer( CPixelToImageLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetImageHeight(int value) { Layer<CPixelToImageLayer>()->SetImageHeight(value); }
	int GetImageHeight() const { return Layer<CPixelToImageLayer>()->GetImageHeight(); }
	void SetImageWidth(int value) { Layer<CPixelToImageLayer>()->SetImageWidth(value); }
	int GetImageWidth() const { return Layer<CPixelToImageLayer>()->GetImageWidth(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "PixelToImage" );
		return pyConstructor( py::cast(this), 0, 0 );
	}
};

class CPyImageResizeLayer : public CPyLayer {
public:
	explicit CPyImageResizeLayer( CImageResizeLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetDeltas(const py::list& deltas)
	{
		Layer<CImageResizeLayer>()->SetDelta(CImageResizeLayer::IS_Left, deltas[0].cast<int>());
		Layer<CImageResizeLayer>()->SetDelta(CImageResizeLayer::IS_Right, deltas[1].cast<int>());
		Layer<CImageResizeLayer>()->SetDelta(CImageResizeLayer::IS_Top, deltas[2].cast<int>());
		Layer<CImageResizeLayer>()->SetDelta(CImageResizeLayer::IS_Bottom, deltas[3].cast<int>());
	}
	py::list GetDeltas() const
	{
	        py::list list;
		list.append(Layer<CImageResizeLayer>()->GetDelta(CImageResizeLayer::IS_Left));
		list.append(Layer<CImageResizeLayer>()->GetDelta(CImageResizeLayer::IS_Right));
		list.append(Layer<CImageResizeLayer>()->GetDelta(CImageResizeLayer::IS_Top));
		list.append(Layer<CImageResizeLayer>()->GetDelta(CImageResizeLayer::IS_Bottom));
		return list;
	}

	void SetDefaultValue(float value) { Layer<CImageResizeLayer>()->SetDefaultValue(value); }
	float GetDefaultValue() const { return Layer<CImageResizeLayer>()->GetDefaultValue(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ImageResize" );
		return pyConstructor( py::cast(this), 0 );
	}
};

void InitializeImageConversionLayer( py::module& m )
{
	py::class_<CPyImageToPixelLayer, CPyLayer>(m, "ImageToPixel")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyImageToPixelLayer( *layer.Layer<CImageToPixelLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, const CPyLayer& layer2, int outputNumber1, int outputNumber2 ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CImageToPixelLayer> accuracy = new CImageToPixelLayer( mathEngine );
			accuracy->SetName( FindFreeLayerName( dnn, "ImageToPixel", name ).c_str() );
			dnn.AddLayer( *accuracy );
			accuracy->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			accuracy->Connect( 1, layer2.BaseLayer(), outputNumber2 );
			return new CPyImageToPixelLayer( *accuracy, layer1.MathEngineOwner() );
		}) )
	;

	py::class_<CPyPixelToImageLayer, CPyLayer>(m, "PixelToImage")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyPixelToImageLayer( *layer.Layer<CPixelToImageLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, const CPyLayer& layer2, int outputNumber2, int height, int width ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CPixelToImageLayer> conv = new CPixelToImageLayer( mathEngine );
			conv->SetImageHeight(height);
			conv->SetImageWidth(width);
			conv->SetName( FindFreeLayerName( dnn, "PixelToImage", name ).c_str() );
			dnn.AddLayer( *conv );
			conv->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			conv->Connect( 1, layer2.BaseLayer(), outputNumber2 );
			return new CPyPixelToImageLayer( *conv, layer1.MathEngineOwner() );
		}) )
		.def( "get_height", &CPyPixelToImageLayer::GetImageHeight, py::return_value_policy::reference )
		.def( "set_height", &CPyPixelToImageLayer::SetImageHeight, py::return_value_policy::reference )
		.def( "get_width", &CPyPixelToImageLayer::GetImageWidth, py::return_value_policy::reference )
		.def( "set_width", &CPyPixelToImageLayer::SetImageWidth, py::return_value_policy::reference )
	;

	py::class_<CPyImageResizeLayer, CPyLayer>(m, "ImageResize")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyImageResizeLayer( *layer.Layer<CImageResizeLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, int left, int right, int top, int bottom, float defaultValue ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CImageResizeLayer> resize = new CImageResizeLayer( mathEngine );
			resize->SetDelta(CImageResizeLayer::IS_Left, left);
			resize->SetDelta(CImageResizeLayer::IS_Right, right);
			resize->SetDelta(CImageResizeLayer::IS_Top, top);
			resize->SetDelta(CImageResizeLayer::IS_Bottom, bottom);
			resize->SetDefaultValue(defaultValue);
			resize->SetName( FindFreeLayerName( dnn, "ImageResize", name ).c_str() );
			dnn.AddLayer( *resize );
			resize->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return new CPyImageResizeLayer( *resize, layer1.MathEngineOwner() );
		}) )
		.def( "get_deltas", &CPyImageResizeLayer::GetDeltas, py::return_value_policy::reference )
		.def( "set_deltas", &CPyImageResizeLayer::SetDeltas, py::return_value_policy::reference )
		.def( "get_default_value", &CPyImageResizeLayer::GetDefaultValue, py::return_value_policy::reference )
		.def( "set_default_value", &CPyImageResizeLayer::SetDefaultValue, py::return_value_policy::reference )
	;
}
