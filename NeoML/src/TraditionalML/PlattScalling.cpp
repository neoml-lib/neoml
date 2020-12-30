/* Copyright © 2017-2020 ABBYY Production LLC

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

#include <NeoML/TraditionalML/PlattScalling.h>
#include <NeoML/TraditionalML/FloatVector.h>
#include <NeoML/TraditionalML/LinearClassifier.h>

namespace NeoML {

// The likelihood function that should be maximized in Platt method (see https://www.csie.ntu.edu.tw/~cjlin/papers/plattprob.pdf)
// sum( t[i] * log( P[i] ) + ( 1 - t[i] ) * log ( 1 - P[i] ) ), P[i] = 1 / 1 + exp( A * output[i] + B )
class CLikelihoodFunction {
public:
	CLikelihoodFunction( const IProblem& trainingClassificationData, const CArray<double>& output );

	// Calculates the function value
	double CalculateValue( const CSigmoid& coefficients ) const;
	
	// Sets the argument and recalculates the gradient and the hessian
	void SetArgument( const CSigmoid& coefficients );

	// Gets the function value
	double Value() const { return value; }

	// Gets the function gradient
	void Gradient( double& g0, double& g1 ) const;

	// Multiplies a vector by the function hessian
	void ProductHessian( double v0, double v1, double& h0, double& h1 ) const;

private:
	CArray<double> values; // the classification results
	CArray<double> weights; // vector weights
	CArray<double> t; // auxiliary coefficients for the function
	double gradient[2]; // gradient
	double hessian[2][2]; // hessian
	double value; // function value
};

CLikelihoodFunction::CLikelihoodFunction( const IProblem& trainingClassificationData, const CArray<double>& output )
{
	output.CopyTo( values );

	double posCount = 0;
	double negCount = 0;
	weights.SetBufferSize( trainingClassificationData.GetVectorCount() );
	for( int i = 0; i < trainingClassificationData.GetVectorCount(); i++ ) {
		weights.Add( trainingClassificationData.GetVectorWeight( i ) );
		if( trainingClassificationData.GetBinaryClass( i ) > 0 ) {
			posCount += weights[i];
		} else {
			negCount += weights[i];
		}
	}

	const double tPositive = ( posCount + 1.0 ) / ( posCount + 2.0 );
	const double tNegative = 1 / ( negCount + 2.0 );
	t.SetBufferSize( trainingClassificationData.GetVectorCount() );
	for( int i = 0; i < trainingClassificationData.GetVectorCount(); i++ ) {
		if( trainingClassificationData.GetBinaryClass( i ) > 0 ) {
			t.Add( tPositive );
		} else {
			t.Add( tNegative );
		}
	}
}

double CLikelihoodFunction::CalculateValue( const CSigmoid& coefficients ) const
{
	double result = 0;
	for( int i = 0; i < values.Size(); i++ ) {
		double temp = values[i] * coefficients.A + coefficients.B;
		if( temp >= 0 ) {
			result += weights[i] * ( t[i] * temp + log( 1 + exp( -temp ) ) );
		} else {
			result += weights[i] * ( ( t[i] - 1 ) * temp + log( 1 + exp( temp ) ) );
		}
	}
	return result;
}

void CLikelihoodFunction::SetArgument( const CSigmoid& coefficients )
{
	// Update the function value, gradient and hessian
	hessian[0][0] = 1e-12;
	hessian[1][1] = 1e-12;
	hessian[1][0] = 0.0;
	hessian[0][1] = 0.0;

	gradient[0] = 0;
	gradient[1] = 0;

	value = 0;
	for( int i = 0; i < values.Size(); i++ ) {
		double temp = values[i] * coefficients.A + coefficients.B;
		double p = 0.0;
		double q = 0.0;
		if( temp >= 0 ) {
			value += weights[i] * ( t[i] * temp + log( 1 + exp( -temp ) ) );
			p = exp( -temp ) / ( 1.0 + exp( -temp ) );
			q = 1.0 / ( 1.0 + exp( -temp ) );
		} else {
			value += weights[i] * ( ( t[i] - 1 ) * temp + log( 1 + exp( temp ) ) );
			p = 1.0 / ( 1.0 + exp( temp ) );
			q = exp( temp ) / ( 1.0 + exp( temp ) );
		}

		hessian[0][0]+= weights[i] * values[i] * values[i] * p * q;
		hessian[1][1]+= weights[i] * p * q;
		hessian[1][0]+= weights[i] * values[i] * p * q;

		gradient[0]+= weights[i] * values[i] * ( t[i] - p );
		gradient[1]+= weights[i] * ( t[i] - p );
	}
}

void CLikelihoodFunction::Gradient( double& g0, double& g1 ) const
{
	g0 = gradient[0];
	g1 = gradient[1];
}

void CLikelihoodFunction::ProductHessian( double v0, double v1, double& h0, double& h1 ) const
{
	double det = hessian[0][0] * hessian[1][1] - hessian[1][0] * hessian[1][0];
	h0 = -( hessian[1][1] * v0 - hessian[1][0] * v1 ) / det;
	h1 = -( -hessian[1][0] * v0 + hessian[0][0] * v1 ) / det;
}

void CalcSigmoidCoefficients( const IProblem& trainingClassificationData, const CArray<double>& output,
	CSigmoid& coefficients )
{
	double posCount = 0;
	double negCount = 0;
	for( int i = 0; i < trainingClassificationData.GetVectorCount(); i++ ) {
		if( trainingClassificationData.GetBinaryClass( i ) > 0 ) {
			posCount += trainingClassificationData.GetVectorWeight( i );
		} else {
			negCount += trainingClassificationData.GetVectorWeight( i );
		}
	}

	// Use the Newton method with backtracking
	const double eps = 1e-5;

	coefficients.A = 0.0;
	coefficients.B = log( ( negCount + 1.0 ) / ( posCount + 1.0 ) );

	CLikelihoodFunction function( trainingClassificationData, output );
	
	for( int i = 0; i < 100; i++ ) {
		function.SetArgument( coefficients );
		double value = function.Value();
		double g0 = 0;
		double g1 = 0;
		function.Gradient( g0, g1 );
		if( fabs( g0 ) < eps && fabs( g1 ) < eps ) {
			break;
		}

		double h0 = 0;
		double h1 = 0;
		function.ProductHessian( g0, g1, h0, h1 );
		double length = g0 * h0 + g1 * h1;

		double step = 1;
		while( step >= 1e-10 ) {
			CSigmoid newCoefficients = coefficients;
			newCoefficients.A += h0 * step;
			newCoefficients.B += h1 * step;
			double newValue = function.CalculateValue( newCoefficients );

			if( newValue < value + 0.0001 * step * length ) {
				coefficients = newCoefficients;
				break;
			}

			step = step / 2;
		}
	}

	if( !coefficients.IsValid() ) {
		// The Platt method should not dramatically change the classification results; if that is the case, ignore it
		coefficients.A = -1.;
		coefficients.B = 0.;
	}
}

void CalcSigmoidCoefficients( const CCrossValidationResult& crossValidationResult, CSigmoid& coefficients )
{
	NeoAssert( crossValidationResult.Problem != 0 );
	NeoAssert( !crossValidationResult.Models.IsEmpty() );
	NeoAssert( !crossValidationResult.Results.IsEmpty() );
	NeoAssert( !crossValidationResult.Success.IsEmpty() );
	NeoAssert( dynamic_cast<ILinearBinaryModel*>( crossValidationResult.Models.First().Ptr() ) != 0 );

	CArray<double> output;
	output.SetSize( crossValidationResult.Problem->GetVectorCount() );
	CSparseFloatMatrixDesc matrix = crossValidationResult.Problem->GetMatrix();
	CSparseFloatVectorDesc vector;

	for( int i = 0; i < crossValidationResult.Results.Size(); i++ ) {
		const int modelIndex = crossValidationResult.ModelIndex[i];
		CPtr<ILinearBinaryModel> model = dynamic_cast<ILinearBinaryModel*>(
			crossValidationResult.Models[modelIndex].Ptr() );
		matrix.GetRow( i, vector );
		const double distance = LinearFunction( model->GetPlane(), vector );
		output[i] = distance;
	}

	CalcSigmoidCoefficients( *crossValidationResult.Problem, output, coefficients );
}

} // namespace NeoML
