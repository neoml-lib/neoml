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

#include <NeoML/Dnn/Layers/Onnx/OnnxLayers.h>

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
REGISTER_NEOML_LAYER( COnnxShapeToBlobLayer, "NeoMLDnnOnnxShapeToBlobLayer" )
REGISTER_NEOML_LAYER( COnnxSliceLayer, "NeoMLDnnOnnxSliceLayer" )
REGISTER_NEOML_LAYER( COnnxSourceHelper, "NeoMLDnnOnnxSourceHelper" )
REGISTER_NEOML_LAYER( COnnxSplitLayer, "NeoMLDnnOnnxSplitLayer" )
REGISTER_NEOML_LAYER( COnnxTransformHelper, "NeoMLDnnOnnxTransformHelper" )
REGISTER_NEOML_LAYER( COnnxTransposeHelper, "NeoMLDnnOnnxTransposeHelper" )

}

} // namespace NeoML
