/* Copyright © 2017-2023 ABBYY

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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/TraditionalML/FloatVector.h>
#include <NeoML/TraditionalML/Problem.h>

namespace NeoML {

// An arbitrary function interface
// Used for optimization problems and error functions
class CFunction {
public:
	virtual ~CFunction() = default;

	// The problem dimensions
	virtual int NumberOfDimensions() const = 0;

	// Sets the argument for the subsequent calculations
	// Used for intermediate calculations for the function value, gradient
	// or multiplying hessian by a vector
	virtual void SetArgument( const CFloatVector& x ) = 0;

	// The function value
	virtual double Value() const = 0;

	// The Evaluate function that is a wrapper over SetArgument and Value
	double Evaluate( const CFloatVector& x );
};

// The interface for a function that supports gradient calculation
class CFunctionWithGradient : public CFunction {
public:
	// Function gradient
	virtual CFloatVector Gradient() const = 0;
};

// The interface for a function that supports gradient and hessian calculation
class CFunctionWithHessian : public CFunctionWithGradient {
public:
	// The product of the function hessian by a given vector
	virtual CFloatVector HessianProduct( const CFloatVector& ) = 0;
};

//------------------------------------------------------------------------------------------------------------

struct CFunctionWithHessianState;

enum class THessianFType {
	SquaredHinge, L2Regression, LogRegression, SmoothedHinge
};

// Function that supports gradient and hessian calculation in multiple threads
class IMultiThreadFunctionWithHessianImpl : public CFunctionWithHessian {
public:
	~IMultiThreadFunctionWithHessianImpl() override;

	// The product of the function hessian by a given vector
	CFloatVector HessianProduct( const CFloatVector& ) override final;

	// The CFunctionWithHessian class methods:
	int NumberOfDimensions() const override final;
	double Value() const override final;
	CFloatVector Gradient() const override final;
	// Sets the argument for the subsequent calculations
	void SetArgument( const CFloatVector& ) override final;

protected:
	IMultiThreadFunctionWithHessianImpl( const IProblem&,
		double errorWeight, float l1Coeff, int threadCount, THessianFType );
	IMultiThreadFunctionWithHessianImpl( const IRegressionProblem&,
		double errorWeight, double p, float l1Coeff, int threadCount, THessianFType );

	CFunctionWithHessianState* const FS; // Function State (internal)
};

//------------------------------------------------------------------------------------------------------------
// Main loss functions

// For support-vector machine with a squared hinge loss function:
class NEOML_API CSquaredHinge : public IMultiThreadFunctionWithHessianImpl {
public:
	CSquaredHinge( const IProblem& problem, double errorWeight, float l1Coeff, int threadCount ) :
		IMultiThreadFunctionWithHessianImpl( problem, errorWeight, l1Coeff, threadCount, THessianFType::SquaredHinge )
	{}
};

//------------------------------------------------------------------------------------------------------------

// Loss function for a regression problem
class NEOML_API CL2Regression : public IMultiThreadFunctionWithHessianImpl {
public:
	CL2Regression( const IRegressionProblem& problem, double errorWeight, double p, float l1Coeff, int threadCount ) :
		IMultiThreadFunctionWithHessianImpl( problem, errorWeight, p, l1Coeff, threadCount, THessianFType::L2Regression )
	{}
};

//------------------------------------------------------------------------------------------------------------

// Logistic regression function
class NEOML_API CLogRegression: public IMultiThreadFunctionWithHessianImpl {
public:
	CLogRegression( const IProblem& problem, double errorWeight, float l1Coeff, int threadCount ) :
		IMultiThreadFunctionWithHessianImpl( problem, errorWeight, l1Coeff, threadCount, THessianFType::LogRegression )
	{}
};

//------------------------------------------------------------------------------------------------------------

// Smoothed hinge function
class NEOML_API CSmoothedHinge : public IMultiThreadFunctionWithHessianImpl {
public:
	CSmoothedHinge( const IProblem& problem, double errorWeight, float l1Coeff, int threadCount ) :
		IMultiThreadFunctionWithHessianImpl( problem, errorWeight, l1Coeff, threadCount, THessianFType::SmoothedHinge )
	{}
};

//------------------------------------------------------------------------------------------------------------

// inline implementation
inline double CFunction::Evaluate( const CFloatVector& x )
{
	SetArgument( x );
	return Value();
}

} // namespace NeoML
