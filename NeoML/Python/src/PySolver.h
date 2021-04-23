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

#include "PyMathEngine.h"

class CPySolver {
public:
	CPySolver( CDnnSolver& _solver, CPyMathEngineOwner& _mathEngineOwner ) :
		solver( &_solver ), mathEngineOwner( &_mathEngineOwner ) {}

	std::string GetClassName() const;

	CPyMathEngineOwner& MathEngineOwner() const { return *mathEngineOwner; }

	CDnnSolver& BaseSolver() const { return *solver; }

	template<class T>
	T* Solver() const { return dynamic_cast<T*>( solver.Ptr() ); }

	void Train() { solver->Train(); }

	void Reset() { solver->Reset(); }

	float GetLearningRate() const { return solver->GetLearningRate(); }
	void SetLearningRate( float learningRate ) { solver->SetLearningRate( learningRate ); }
	float GetL2Regularization() const { return solver->GetL2Regularization(); }
	void SetL2Regularization( float regularization ) { solver->SetL2Regularization( regularization ); }
	float GetL1Regularization() const { return solver->GetL1Regularization(); }
	void SetL1Regularization( float regularization ) { solver->SetL1Regularization( regularization ); }
	float GetMaxGradientNorm() const { return solver->GetMaxGradientNorm(); }
	void SetMaxGradientNorm( float maxGradientNorm ) { solver->SetMaxGradientNorm( maxGradientNorm ); }

private:
	CPtr<CPyMathEngineOwner> mathEngineOwner;
	CPtr<CDnnSolver> solver;
};

void InitializeSolver( py::module& m );
