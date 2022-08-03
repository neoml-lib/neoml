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

#include "../LayerOperator.h"

namespace NeoOnnx {

// Base class for operators which implement activation functions from NeoML
class CActivationOperatorBase : public CLayerOperator {
public:
	CActivationOperatorBase( const onnx::NodeProto& onnxNode, int opsetVersion,
		TActivationFunction activation );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;

private:
	// Activation function which is applied to the input by this operator
	const TActivationFunction activation;
};

//---------------------------------------------------------------------------------------------------------------------
// Operators which are implementing activation functions

// Abs operator
class CAbsOperator : public CActivationOperatorBase {
public:
	CAbsOperator( const onnx::NodeProto& abs, int opsetVersion );
};

// Clip operator
class CClipOperator : public CActivationOperatorBase {
public:
	CClipOperator( const onnx::NodeProto& clip, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

// Elu operator
class CEluOperator : public CActivationOperatorBase {
public:
	CEluOperator( const onnx::NodeProto& elu, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

// HardSigmoid operator
class CHardSigmoidOperator : public CActivationOperatorBase {
public:
	CHardSigmoidOperator( const onnx::NodeProto& hardSigmoid, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

// LeakyRelu operator
class CLeakyReluOperator : public CActivationOperatorBase {
public:
	CLeakyReluOperator( const onnx::NodeProto& leakyRelu, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

// Pow operator
class CPowOperator : public CActivationOperatorBase {
public:
	CPowOperator( const onnx::NodeProto& pow, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

// Relu operator
class CReluOperator : public CActivationOperatorBase {
public:
	CReluOperator( const onnx::NodeProto& relu, int opsetVersion );
};

// Sigmoid operator
class CSigmoidOperator : public CActivationOperatorBase {
public:
	CSigmoidOperator( const onnx::NodeProto& sigmoid, int opsetVersion );
};

// Sqrt operator
class CSqrtOperator : public CActivationOperatorBase {
public:
	CSqrtOperator( const onnx::NodeProto& sqrt, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

// Tanh operator
class CTanhOperator : public CActivationOperatorBase {
public:
	CTanhOperator( const onnx::NodeProto& tanh, int opsetVersion );
};

// Exp operator
class CExpOperator : public CActivationOperatorBase {
public:
	CExpOperator( const onnx::NodeProto& exp, int opsetVersion );
};

// Log operator
class CLogOperator : public CActivationOperatorBase {
public:
	CLogOperator( const onnx::NodeProto& log, int opsetVersion );
};

// Erf operator
class CErfOperator : public CActivationOperatorBase {
public:
	CErfOperator( const onnx::NodeProto& erf, int opsetVersion );
};

// Neg operator
class CNegOperator : public CActivationOperatorBase {
public:
	CNegOperator( const onnx::NodeProto& neg, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
};

} // namespace NeoOnnx

