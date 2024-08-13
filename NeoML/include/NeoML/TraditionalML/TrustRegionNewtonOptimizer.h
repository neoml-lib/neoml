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

// The Newton method for finding a minimum value in the trust region
// See the paper:
// Chih-Jen Lin, Ruby C. Weng, and S. Sathiya Keerthi.
// "Trust region Newton method for large-scale logistic regression."
// "Journal of Machine Learning Research", 9:627-650, 2008.
// URL http://www.csie.ntu.edu.tw/~cjlin/papers/logistic.pdf.

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/TraditionalML/FloatVector.h>
#include <NeoML/TraditionalML/Problem.h>
#include <NeoML/TraditionalML/Function.h>

namespace NeoML {

// The class implements the search for a function minimum using conjugate gradient method with a trust region
class NEOML_API CTrustRegionNewtonOptimizer {
public:
	// function is the function to optimize 
	// tolerance specifies the stop criterion (to stop, the gradient should not be greater than the starting gradient, which equals this value)
	// maxIterations is the maximum number of algorithm iterations
	CTrustRegionNewtonOptimizer(CFunctionWithHessian *function, double tolerance = 0.01, int maxIterations = 1000);

	// Sets the initial approximation
	void SetInitialArgument(const CFloatVector& initialArgument) { currentArgument = initialArgument; }
	// Starts the optimization cycle
	void Optimize();
	// Retrieves the optimal value
	CFloatVector GetOptimalArgument() { return currentArgument; }
	// Sets a text stream for logging processing
	// By default logging is off (set to null to turn off)
	void SetLog(CTextStream* newLog) { log = newLog; }

private:
	double tolerance; // the stop criterion
	int maxIterations; // the iteration limit
	CFunctionWithHessian *function; // the function to optimize
	CFloatVector currentArgument; // the current answer
	CTextStream* log; // the logging stream

	int conjugateGradientSearch(double trustRegionSize, const CFloatVector& gradient,
		CFloatVector& shift, CFloatVector& residue);
};

} // namespace NeoML
