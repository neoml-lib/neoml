/* Copyright © 2017-2022 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/Onnx/OnnxCastLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxConcatLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxConstantOfShapeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxConvTransposeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxExpandLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxGatherLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxNonZeroLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxOneHotLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxRangeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxReshapeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxResizeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxShapeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxSliceLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxSourceHelper.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxSplitLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxTransformHelper.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxTransposeHelper.h>
#include <NeoML/Dnn/Layers/Onnx/ShapeToBlobLayer.h>

namespace NeoML {

namespace {

REGISTER_NEOML_LAYER( COnnxCastLayer, "NeoMLDnnOnnxCastLayer" )
REGISTER_NEOML_LAYER( COnnxConcatLayer, "NeoMLDnnOnnxConcatLayer" )
REGISTER_NEOML_LAYER( COnnxConstantOfShapeLayer, "NeoMLDnnOnnxConstantOfShapeLayer" )
REGISTER_NEOML_LAYER( COnnxConvTransposeLayer, "NeoMLDnnOnnxConvTransposeLayer" )
REGISTER_NEOML_LAYER( COnnxEltwiseLayer, "NeoMLDnnOnnxEltwiseLayer" )
REGISTER_NEOML_LAYER( COnnxExpandLayer, "NeoMLDnnOnnxExpandLayer" )
REGISTER_NEOML_LAYER( COnnxGatherLayer, "NeoMLDnnOnnxGatherLayer" )
REGISTER_NEOML_LAYER( COnnxNonZeroLayer, "NeoMLDnnOnnxNonZeroLayer" )
REGISTER_NEOML_LAYER( COnnxOneHotLayer, "NeoMLDnnOnnxOneHotLayer" )
REGISTER_NEOML_LAYER( COnnxRangeLayer, "NeoMLDnnOnnxRangeLayer" )
REGISTER_NEOML_LAYER( COnnxReshapeLayer, "NeoMLDnnOnnxReshapeLayer" )
REGISTER_NEOML_LAYER( COnnxResizeLayer, "NeoMLDnnOnnxResizeLayer" )
REGISTER_NEOML_LAYER( COnnxShapeLayer, "NeoMLDnnOnnxShapeLayer" )
REGISTER_NEOML_LAYER( COnnxSliceLayer, "NeoMLDnnOnnxSliceLayer" )
REGISTER_NEOML_LAYER( COnnxSourceHelper, "NeoMLDnnOnnxSourceHelper" )
REGISTER_NEOML_LAYER( COnnxSplitLayer, "NeoMLDnnOnnxSplitLayer" )
REGISTER_NEOML_LAYER( COnnxTransformHelper, "NeoMLDnnOnnxTransformHelper" )
REGISTER_NEOML_LAYER( COnnxTransposeHelper, "NeoMLDnnOnnxTransposeHelper" )
REGISTER_NEOML_LAYER( CShapeToBlobLayer, "NeoMLDnnShapeToBlobLayer" )

}

} // namespace NeoML
