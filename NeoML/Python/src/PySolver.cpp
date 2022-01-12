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

#include "PySolver.h"
#include "PyMathEngine.h"

std::string CPySolver::GetClassName() const
{
	if( Solver<CDnnNesterovGradientSolver>() != 0 ) {
		return "NesterovGradient";
	}
	if( Solver<CDnnSimpleGradientSolver>() != 0 ) {
		return "SimpleGradient";
	}
	if( Solver<CDnnAdaptiveGradientSolver>() != 0 ) {
		return "AdaptiveGradient";
	}
	assert( false );
	return "";
}

//------------------------------------------------------------------------------------------------------------

class CPySimpleGradientSolver : public CPySolver {
public:
	CPySimpleGradientSolver( CDnnSimpleGradientSolver& solver, CPyMathEngineOwner& owner ) :
		CPySolver( solver, owner ) {}

	float GetMomentDecayRate() const { return Solver<CDnnSimpleGradientSolver>()->GetMomentDecayRate(); }
	void SetMomentDecayRate( float decayRate ) { Solver<CDnnSimpleGradientSolver>()->SetMomentDecayRate( decayRate ); }
};

//------------------------------------------------------------------------------------------------------------

class CPyAdaptiveGradientSolver : public CPySolver {
public:
	CPyAdaptiveGradientSolver( CDnnAdaptiveGradientSolver& solver, CPyMathEngineOwner& owner ) :
		CPySolver( solver, owner ) {}

	float GetMomentDecayRate() const { return Solver<CDnnAdaptiveGradientSolver>()->GetMomentDecayRate(); }
	void SetMomentDecayRate( float decayRate ) { Solver<CDnnAdaptiveGradientSolver>()->SetMomentDecayRate( decayRate ); }

	float GetSecondMomentDecayRate() const { return Solver<CDnnAdaptiveGradientSolver>()->GetSecondMomentDecayRate(); }
	void SetSecondMomentDecayRate( float decayRate ) { Solver<CDnnAdaptiveGradientSolver>()->SetSecondMomentDecayRate( decayRate ); }

	float GetEpsilon() const { return Solver<CDnnAdaptiveGradientSolver>()->GetEpsilon(); }
	void SetEpsilon( float newEpsilon ) { Solver<CDnnAdaptiveGradientSolver>()->SetEpsilon( newEpsilon ); }

	bool GetAmsGrad() const { return Solver<CDnnAdaptiveGradientSolver>()->IsAmsGradEnabled(); }
	void SetAmsGrad( bool enable ) { Solver<CDnnAdaptiveGradientSolver>()->EnableAmsGrad( enable ); }

	bool GetDecoupledWeightDecay() const { return Solver<CDnnAdaptiveGradientSolver>()->IsDecoupledWeightDecay(); }
	void SetDecoupledWeightDecay( bool enable ) { Solver<CDnnAdaptiveGradientSolver>()->EnableDecoupledWeightDecay( enable ); }
};

//------------------------------------------------------------------------------------------------------------

class CPyNesterovGradientSolver : public CPySolver {
public:
	CPyNesterovGradientSolver( CDnnNesterovGradientSolver& solver, CPyMathEngineOwner& owner ) :
		CPySolver( solver, owner ) {}

	float GetMomentDecayRate() const { return Solver<CDnnNesterovGradientSolver>()->GetMomentDecayRate(); }
	void SetMomentDecayRate( float decayRate ) { Solver<CDnnNesterovGradientSolver>()->SetMomentDecayRate( decayRate ); }

	float GetSecondMomentDecayRate() const { return Solver<CDnnNesterovGradientSolver>()->GetSecondMomentDecayRate(); }
	void SetSecondMomentDecayRate( float decayRate ) { Solver<CDnnNesterovGradientSolver>()->SetSecondMomentDecayRate( decayRate ); }

	float GetEpsilon() const { return Solver<CDnnNesterovGradientSolver>()->GetEpsilon(); }
	void SetEpsilon( float newEpsilon ) { Solver<CDnnNesterovGradientSolver>()->SetEpsilon( newEpsilon ); }

	bool GetAmsGrad() const { return Solver<CDnnNesterovGradientSolver>()->IsAmsGradEnabled(); }
	void SetAmsGrad( bool enable ) { Solver<CDnnNesterovGradientSolver>()->EnableAmsGrad( enable ); }
};

void InitializeSolver( py::module& m )
{
	py::class_<CPySolver>(m, "Solver")
		.def( "_train", &CPySolver::Train, py::call_guard<py::gil_scoped_release>(), py::return_value_policy::reference )
		.def( "_reset", &CPySolver::Reset, py::return_value_policy::reference )
		.def( "get_learning_rate", &CPySolver::GetLearningRate, py::return_value_policy::reference )
		.def( "set_learning_rate", &CPySolver::SetLearningRate, py::return_value_policy::reference )
		.def( "get_l2", &CPySolver::GetL2Regularization, py::return_value_policy::reference )
		.def( "set_l2", &CPySolver::SetL2Regularization, py::return_value_policy::reference )
		.def( "get_l1", &CPySolver::GetL1Regularization, py::return_value_policy::reference )
		.def( "set_l1", &CPySolver::SetL1Regularization, py::return_value_policy::reference )
		.def( "get_max_gradient_norm", &CPySolver::GetMaxGradientNorm, py::return_value_policy::reference )
		.def( "set_max_gradient_norm", &CPySolver::SetMaxGradientNorm, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPySimpleGradientSolver, CPySolver>(m, "SimpleGradient")
		.def( py::init([]( const CPySolver& solver )
		{
			return new CPySimpleGradientSolver( *solver.Solver<CDnnSimpleGradientSolver>(), solver.MathEngineOwner() );
		}) )
		.def( py::init([]( const CPyMathEngine& mathEngine, float learning_rate, float l1, float l2, float max_grad_norm, float moment_decay_rate )
		{
			CPtr<CDnnSimpleGradientSolver> solver( new CDnnSimpleGradientSolver( mathEngine.MathEngineOwner().MathEngine() ) );
			solver->SetLearningRate(learning_rate);
			solver->SetL2Regularization(l2);
			solver->SetL1Regularization(l1);
			solver->SetMaxGradientNorm(max_grad_norm);

			solver->SetMomentDecayRate(moment_decay_rate);

			return new CPySimpleGradientSolver( *solver, mathEngine.MathEngineOwner() );
		}) )
		.def( "get_moment_decay_rate", &CPySimpleGradientSolver::GetMomentDecayRate, py::return_value_policy::reference )
		.def( "set_moment_decay_rate", &CPySimpleGradientSolver::SetMomentDecayRate, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyAdaptiveGradientSolver, CPySolver>(m, "AdaptiveGradient")
		.def( py::init([]( const CPySolver& solver )
		{
			return new CPyAdaptiveGradientSolver( *solver.Solver<CDnnAdaptiveGradientSolver>(), solver.MathEngineOwner() );
		}) )
		.def( py::init([]( const CPyMathEngine& mathEngine, float learning_rate, float l1, float l2, float max_grad_norm,
			float moment_decay_rate, float second_moment_decay_rate, float epsilon, bool ams_grad, bool decoupled_weight_decay )
		{
			CPtr<CDnnAdaptiveGradientSolver> solver( new CDnnAdaptiveGradientSolver( mathEngine.MathEngineOwner().MathEngine() ) );
			solver->SetLearningRate(learning_rate);
			solver->SetL2Regularization(l2);
			solver->SetL1Regularization(l1);
			solver->SetMaxGradientNorm(max_grad_norm);

			solver->SetMomentDecayRate(moment_decay_rate);
			solver->SetSecondMomentDecayRate(second_moment_decay_rate);
			solver->SetEpsilon(epsilon);
			solver->EnableAmsGrad(ams_grad);
			solver->EnableDecoupledWeightDecay(decoupled_weight_decay);

			return new CPyAdaptiveGradientSolver( *solver, mathEngine.MathEngineOwner() );
		}) )

		.def( "get_moment_decay_rate", &CPyAdaptiveGradientSolver::GetMomentDecayRate, py::return_value_policy::reference )
		.def( "set_moment_decay_rate", &CPyAdaptiveGradientSolver::SetMomentDecayRate, py::return_value_policy::reference )
		.def( "get_second_moment_decay_rate", &CPyAdaptiveGradientSolver::GetSecondMomentDecayRate, py::return_value_policy::reference )
		.def( "set_second_moment_decay_rate", &CPyAdaptiveGradientSolver::SetSecondMomentDecayRate, py::return_value_policy::reference )
		.def( "get_epsilon", &CPyAdaptiveGradientSolver::GetEpsilon, py::return_value_policy::reference )
		.def( "set_epsilon", &CPyAdaptiveGradientSolver::SetEpsilon, py::return_value_policy::reference )
		.def( "get_ams_grad", &CPyAdaptiveGradientSolver::GetAmsGrad, py::return_value_policy::reference )
		.def( "set_ams_grad", &CPyAdaptiveGradientSolver::SetAmsGrad, py::return_value_policy::reference )
		.def( "get_decoupled_weight_decay", &CPyAdaptiveGradientSolver::GetDecoupledWeightDecay, py::return_value_policy::reference )
		.def( "set_decoupled_weight_decay", &CPyAdaptiveGradientSolver::SetDecoupledWeightDecay, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyNesterovGradientSolver, CPySolver>(m, "NesterovGradient")
		.def( py::init([]( const CPySolver& solver )
		{
			return new CPyNesterovGradientSolver( *solver.Solver<CDnnNesterovGradientSolver>(), solver.MathEngineOwner() );
		}) )
		.def( py::init([]( const CPyMathEngine& mathEngine, float learning_rate, float l1, float l2, float max_grad_norm,
			float moment_decay_rate, float second_moment_decay_rate, float epsilon, bool ams_grad )
		{
			CPtr<CDnnNesterovGradientSolver> solver( new CDnnNesterovGradientSolver( mathEngine.MathEngineOwner().MathEngine() ) );
			solver->SetLearningRate(learning_rate);
			solver->SetL2Regularization(l2);
			solver->SetL1Regularization(l1);
			solver->SetMaxGradientNorm(max_grad_norm);

			solver->SetMomentDecayRate(moment_decay_rate);
			solver->SetSecondMomentDecayRate(second_moment_decay_rate);
			solver->SetEpsilon(epsilon);
			solver->EnableAmsGrad(ams_grad);

			return new CPyNesterovGradientSolver( *solver, mathEngine.MathEngineOwner() );
		}) )

		.def( "get_moment_decay_rate", &CPyNesterovGradientSolver::GetMomentDecayRate, py::return_value_policy::reference )
		.def( "set_moment_decay_rate", &CPyNesterovGradientSolver::SetMomentDecayRate, py::return_value_policy::reference )
		.def( "get_second_moment_decay_rate", &CPyNesterovGradientSolver::GetSecondMomentDecayRate, py::return_value_policy::reference )
		.def( "set_second_moment_decay_rate", &CPyNesterovGradientSolver::SetSecondMomentDecayRate, py::return_value_policy::reference )
		.def( "get_epsilon", &CPyNesterovGradientSolver::GetEpsilon, py::return_value_policy::reference )
		.def( "set_epsilon", &CPyNesterovGradientSolver::SetEpsilon, py::return_value_policy::reference )
		.def( "get_ams_grad", &CPyNesterovGradientSolver::GetAmsGrad, py::return_value_policy::reference )
		.def( "set_ams_grad", &CPyNesterovGradientSolver::SetAmsGrad, py::return_value_policy::reference )
	;
}
