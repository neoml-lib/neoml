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

// The method has been described in
// Chih-Jen Lin, Ruby C. Weng, and S. Sathiya Keerthi. 
// "Trust region Newton method for large-scale logistic regression."
// "Journal of Machine Learning Research", 9:627-650, 2008.
// URL http://www.csie.ntu.edu.tw/~cjlin/papers/logistic.pdf.

#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/TrustRegionNewtonOptimizer.h>

namespace NeoML {

CTrustRegionNewtonOptimizer::CTrustRegionNewtonOptimizer(CFunctionWithHessian *function, double tolerance, int maxIterations) :
	tolerance(tolerance),
	maxIterations(maxIterations),
	function(function),		
	log(0)
{
}

void CTrustRegionNewtonOptimizer::Optimize()
{
	// The threshold values for the difference between function increment and its quadratic approximation on each step
	double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;
	// The modification parameters for the trust region trustRegionSize:
	double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

	// If the initial approximation is not set, use the coordinate system origin
	if(currentArgument.IsNull()) {
		currentArgument = CFloatVector(function->NumberOfDimensions());
		currentArgument.Nullify();
	}
	// Set the initial approximation
	function->SetArgument(currentArgument);

	double value = function->Value(); // function value
	CFloatVector gradient = function->Gradient(); // function gradient
	CFloatVector shift(function->NumberOfDimensions()); // solution shift found by the conjugate gradient method 
		// for the quadratic approximation on each step
	CFloatVector residue(function->NumberOfDimensions()); // the anti-gradient of the quadratic approximation
		// in its estimated minimum currentArgument + shift: 
		// residue = -gradient - Hessian*shift;

	double gradientNorm = gradient.Norm(); // the current gradient norm
	// Set the initial trust region size
	double trustRegionSize = gradientNorm; // trust region size
	double initialGradientNorm = gradientNorm; // the initial gradient norm
	
	for(int i = 1; i <= maxIterations; ) {
		// Search for the estimated minimum of the quadratic approximation for the target function
		// in a sphere of trustRegionSize diameter
		// The quadratic approximation: value = gradient*shift + shift*Hessian*shift / 2
		// The quadratic approximation anti-gradient: residue = -gradient - Hessian*shift 
		int cgIterations = conjugateGradientSearch(trustRegionSize, gradient, shift, residue);
		// The new approximation on the next step
		CFloatVector newArgument = currentArgument + shift;
		double gradient_shift = DotProduct(gradient, shift);
		// Calculate the predicted target function value reduction using the quadratic approximation
		double predictedReduction = -0.5 * (gradient_shift - DotProduct(shift, residue));
		// Calculate the actual target function value reduction		
		function->SetArgument(newArgument);
		double newValue = function->Value();
		double actualReduction = value - newValue;
		// Set the trust region size after the first iteration results
		// shiftNorm < trustRegionSize if we are in the neighborhood of the minimum
		double shiftNorm = shift.Norm();
		if(i == 1) {
			trustRegionSize = min(trustRegionSize, shiftNorm);
		}
		// Calculate the upper limit to the trust region size
		double alpha;
		if(actualReduction < -gradient_shift) {
			alpha = max(sigma1, 0.5 * (gradient_shift / (actualReduction + gradient_shift))) * shiftNorm;
		} else {
			alpha = sigma3 * shiftNorm;
		}
		// Update the trust region size using the predicted and the actual function value reduction
		if (actualReduction < eta0 * predictedReduction)
			trustRegionSize = min(alpha, sigma2 * trustRegionSize);
		else if (actualReduction < eta1 * predictedReduction)
			trustRegionSize = max(sigma1 * trustRegionSize, min(alpha, sigma2 * trustRegionSize));
		else if (actualReduction < eta2 * predictedReduction)
			trustRegionSize = max(sigma1 * trustRegionSize, min(alpha, sigma3 * trustRegionSize));
		else
			trustRegionSize = max(trustRegionSize, min(alpha, sigma3 * trustRegionSize));
		// Log the current iteration parameters
		if(log != 0) {
			*log << "iter = " << i << ", actual = " << actualReduction << ", predicted = " << predictedReduction 
				<< ", trust region size = " << trustRegionSize << ", value = " << value << ", gradient norm = " << gradientNorm
				<< ", shift norm = " << shiftNorm
				<< ", conjugate gradient iterations = " << cgIterations << "\n";
		}
		// Accept the new approximation to the minimum
		if(actualReduction > eta0 * predictedReduction) {
			i += 1;
			currentArgument = newArgument;
			value = newValue;
			gradient = function->Gradient();
			gradientNorm = gradient.Norm();
			// The stop criterion on the current and the initial gradient norm
			if(gradientNorm <= tolerance * initialGradientNorm) {
				break;
			}
		// Reject the new approximation and return to the previous one
		} else {
			function->SetArgument(currentArgument);
		}
		// Emergency stop criteria
		if(value < -1.0e+32) {
			if(log != 0) {
				*log << "WARNING: value < -1.0e+32\n";
			}
			break;
		}
		if(actualReduction <= 0 && predictedReduction <= 0) {
			if(log != 0) {
				*log << "WARNING: actual reduction and predicted reduction <= 0\n";
			}
		}
		if(fabs(actualReduction) <= 1.0e-12*fabs(value) &&
		    fabs(predictedReduction) <= 1.0e-12*fabs(value))
		{
			if(log != 0) {
				*log << "WARNING: actual reduction and predicted reduction are too small\n";
			}
			break;
		}
	}
}

// The search for estimated minimum of the quadratic approximation of the target function 
// in a sphere of trustRegionSize diameter
// gradient is the gradient in the currentArgument point
// shift is the distance from the current point to the minimum of the quadratic approximation
// residue is the anti-gradient in the solution point. May not be 0 because we are looking in a limited region
//
// The solution is represented as decomposition into conjugate directions:
// shift = sum(alpha * conjugateVector)

int CTrustRegionNewtonOptimizer::conjugateGradientSearch(double trustRegionSize, 
	const CFloatVector& gradient, CFloatVector& shift, CFloatVector& residue)
{
	// Maximum number of iterations
	const int maxAllowedIterations = 10000;

	// The minimum positive double value that is not 0 (all smaller values are considered equivalent to 0)
	const double Epsilon = 1e-40;

	CFloatVector conjugateVector; // the vector conjugate with respect to the hessian to all preceding vectors
	CFloatVector conjugateVector_Hessian; // the product of the conjugate vector by the hessian

	shift.Nullify();
	residue.Nullify();
	residue -= gradient;
	conjugateVector = residue;
	double residue2 = DotProduct(residue, residue);

	double cgTolerance = 0.1 * gradient.Norm(); // the approximate solution accuracy

	int iteration = 0;
	for(;;)	{
		if(sqrt(residue2) <= cgTolerance) {
			// We have found a solution of required accuracy
			break;
		}
		iteration += 1;
		
		// Avoid loops: check that the iterations limit has not been reached
		NeoAssert( iteration < maxAllowedIterations );

		// Calculate the product of the conjugate vector by the hessian
		conjugateVector_Hessian = function->HessianProduct(conjugateVector);
		// Calculate the next coefficient of the decomposition into conjugate vectors
		// It should lead to a conditional local minimum when moving along the conjugate direction
		// That is, the gradient in the point found should be orthogonal to the conjugate vector: residue * conjugateVector = 0;
		double dotProduct = DotProduct(conjugateVector, conjugateVector_Hessian);
		if( abs( dotProduct ) > Epsilon ) {
			double alpha = residue2 / dotProduct;
			// Add the next element of conjugate vector decomposition
			CFloatVector oldShift = shift;
			shift.MultiplyAndAdd(conjugateVector, alpha);
			// Check that we are still inside the trust region
			if(shift.Norm() <= trustRegionSize) {
				// Calculate the new anti-gradient
				residue.MultiplyAndAdd(conjugateVector_Hessian, -alpha);
				// Find the new conjugate vector
				double residue2_new = DotProduct(residue, residue);
				conjugateVector *= residue2_new / residue2;
				conjugateVector += residue;
				residue2 = residue2_new;
				continue; // move on
			// We have moved out of the trust region
			} else {
				if(log != 0) {
					*log << "Conjugate gradient search reaches trust region boundary\n";
				}
				// Undo the last step
				shift = oldShift;
			}
		}
		// Find the coefficient before the conjugate vector alpha such that the solution is exactly on the trust region boundary
		// (shift equal to trustRegionSize)
		double conjugateVector2 = DotProduct(conjugateVector, conjugateVector);
		if( conjugateVector2 > Epsilon ) { // if the conjugate vector has degenerated, we cannot find anything better
			double shift_conjugateVector = DotProduct(shift, conjugateVector);
			double shift2 = DotProduct(shift, shift);
			double trustRegionSize2 = trustRegionSize * trustRegionSize;
			// alpha is the positive root of the quadratic equation
			// sqr(shift + alpha*conjugateVector) = sqr(trustRegionSize)
			double alpha = (sqrt(shift_conjugateVector * shift_conjugateVector + 
				conjugateVector2 * (trustRegionSize2 - shift2)) - shift_conjugateVector) / conjugateVector2;
			// Find the new solution and anti-gradient
			shift.MultiplyAndAdd(conjugateVector, alpha);
			residue.MultiplyAndAdd(conjugateVector_Hessian, -alpha);
		}
		break;
	}
	return iteration;
}

} // namespace NeoML
