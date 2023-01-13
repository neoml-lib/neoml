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
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxExpandLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxGatherLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxRangeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxReshapeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxShapeLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxSliceLayer.h>
#include <NeoML/Dnn/Layers/Onnx/ShapeToBlobLayer.h>
#include <NeoML/Dnn/Layers/Onnx/SourceReshaper.h>
#include <NeoML/Dnn/Layers/Onnx/TransformReshaper.h>
#include <NeoML/Dnn/Layers/Onnx/TransposeReshaper.h>

namespace NeoML {

namespace {

REGISTER_NEOML_LAYER( COnnxCastLayer, "NeoMLDnnOnnxCastLayer" )
REGISTER_NEOML_LAYER( COnnxConcatLayer, "NeoMLDnnOnnxConcatLayer" )
REGISTER_NEOML_LAYER( COnnxConstantOfShapeLayer, "NeoMLDnnOnnxConstantOfShapeLayer" )
REGISTER_NEOML_LAYER( COnnxEltwiseLayer, "NeoMLDnnOnnxEltwiseLayer" );
REGISTER_NEOML_LAYER( COnnxExpandLayer, "NeoMLDnnOnnxExpandLayer" )
REGISTER_NEOML_LAYER( COnnxGatherLayer, "NeoMLDnnOnnxGatherLayer" )
REGISTER_NEOML_LAYER( COnnxRangeLayer, "NeoMLDnnOnnxRangeLayer" )
REGISTER_NEOML_LAYER( COnnxReshapeLayer, "NeoMLDnnOnnxReshapeLayer" )
REGISTER_NEOML_LAYER( COnnxShapeLayer, "NeoMLDnnOnnxShapeLayer" )
REGISTER_NEOML_LAYER( COnnxSliceLayer, "NeoMLDnnOnnxSliceLayer" )
REGISTER_NEOML_LAYER( CShapeToBlobLayer, "NeoMLDnnShapeToBlobLayer" )
REGISTER_NEOML_LAYER( CSourceReshaper, "NeoMLDnnSourceReshaper" )
REGISTER_NEOML_LAYER( CTransformReshaper, "NeoMLDnnTransformReshaper" )
REGISTER_NEOML_LAYER( CTransposeReshaper, "NeoMLDnnTransposeReshaper" )

}

} // namespace NeoML
