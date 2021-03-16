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

#include "PyDifferentialEvolution.h"

class CPyParam : public IParam {
public:
	explicit CPyParam( py::object _param ) : param( _param) {}

	py::object Param() const { return param; }

private:
	py::object param;
};

//------------------------------------------------------------------------------------------------------------

class CPyParamTraits : public IParamTraits, public IObject {
public:
	explicit CPyParamTraits(py::object _traits) : traits(_traits) {}

	CFunctionParam GenerateRandom(CRandom&, const CFunctionParam& min, const CFunctionParam& max) const override
	{
		py::object pyMutate = traits.attr("generate");
		CPtr<CPyParam> result = new CPyParam(pyMutate(dynamic_cast<const CPyParam*>(min.Ptr())->Param(), dynamic_cast<const CPyParam*>(max.Ptr())->Param()));
		return result.Ptr();
	}

	CFunctionParam Mutate(CRandom& random, const CFunctionParam& base,
		const CFunctionParam& left, const CFunctionParam& right, double fluctuation,
		const CFunctionParam& min, const CFunctionParam& max) const override
	{
		py::object pyMutate = traits.attr("mutate");
		CPtr<CPyParam> result = new CPyParam(pyMutate(dynamic_cast<const CPyParam*>(base.Ptr())->Param(),
			dynamic_cast<const CPyParam*>(left.Ptr())->Param(), dynamic_cast<const CPyParam*>(right.Ptr())->Param(),
			fluctuation, dynamic_cast<const CPyParam*>(min.Ptr())->Param(), dynamic_cast<const CPyParam*>(max.Ptr())->Param()));
		return result.Ptr();
	}

	bool Less(const CFunctionParam& left, const CFunctionParam& right) const override
	{
		py::object pyLess = traits.attr("less");
		return py::cast<bool>( pyLess(dynamic_cast<const CPyParam*>(left.Ptr())->Param(), dynamic_cast<const CPyParam*>(right.Ptr())->Param()) );
	}

	void Dump(CTextStream&, const CFunctionParam&) const override {}

	py::object ParamTraits() { return traits; }

private:
	py::object traits;
};

//------------------------------------------------------------------------------------------------------------

class CPyFunctionEvaluation : public IFunctionEvaluation {
public:
	explicit CPyFunctionEvaluation(py::object f, py::list lowerBounds, py::list upperBounds, py::list param_traits, py::object result_traits) :
		func(f),
		result(new CPyParamTraits(result_traits))
	{
		for(int i = 0; i < lowerBounds.size(); i++) {
			lowBounds.Add(new CPyParam(lowerBounds[i]));
			highBounds.Add(new CPyParam(upperBounds[i]));
			params.Add(new CPyParamTraits(param_traits[i]));
		}
	}

	int NumberOfDimensions() const override { return params.Size(); }
	const IParamTraits& GetParamTraits(int index) const override { return *params[index]; }
	const IParamTraits& GetResultTraits() const override { return *result; }

	CFunctionParam GetMinConstraint(int index) const override { return lowBounds[index].Ptr(); }
	CFunctionParam GetMaxConstraint(int index) const override { return highBounds[index].Ptr(); }

	void Evaluate(const CArray<CFunctionParamVector>& params, CArray<CFunctionParam>& results) override
	{
		results.DeleteAll();
		results.SetSize(params.Size());
		for(int i = 0; i < params.Size(); i++) {
			results[i] = Evaluate(params[i]);
		}
	}

	CFunctionParam Evaluate(const CFunctionParamVector& param) override
	{
		py::list functionParameters;
		for(int i = 0; i < param.Size(); i++) {
			functionParameters.append(dynamic_cast<const CPyParam*>(param[i].Ptr())->Param());
		}
		py::object pyResult = func(functionParameters);
		CPtr<CPyParam> result = new CPyParam(pyResult);
		return result.Ptr();
	}

private:
	py::object func;
	CObjectArray<CPyParam> lowBounds;
	CObjectArray<CPyParam> highBounds;
	CObjectArray<CPyParamTraits> params;
	CPtr<CPyParamTraits> result;
};

//------------------------------------------------------------------------------------------------------------

class CPyFunctionEvaluationOwner : public IObject {
public:
	CPyFunctionEvaluationOwner( py::object f, py::list lowerBounds, py::list upperBounds, py::list param_traits, py::object result_traits ) :
		evaluation( f, lowerBounds, upperBounds, param_traits, result_traits )
	{}

    CPyFunctionEvaluation& Function() { return evaluation; }

private:
	CPyFunctionEvaluation evaluation;
};                                       

//------------------------------------------------------------------------------------------------------------

class CPyDifferentialEvolutionOwner : public IObject {
public:
	CPyDifferentialEvolutionOwner( IFunctionEvaluation& func, double fluctuation, double cr, int population ) :
		evolution(func, fluctuation, cr, population ) {}  

       CDifferentialEvolution& Evolution() { return evolution; }

private:
	CDifferentialEvolution evolution;
};

//------------------------------------------------------------------------------------------------------------

class CPyDifferentialEvolution {
public:
	CPyDifferentialEvolution( CPyFunctionEvaluationOwner* functionOwner, CPyDifferentialEvolutionOwner* evolutionOwner ) :
		function( functionOwner ), evolution( evolutionOwner ) {} 

	bool BuildNextGeneration() { return evolution->Evolution().BuildNextGeneration(); }

	void Run() { evolution->Evolution().RunOptimization(); }

	py::list GetOptimalVector() const
	{
		CFunctionParamVector vector = evolution->Evolution().GetOptimalVector();
		py::list list;
		for(int i = 0; i < vector.Size(); i++) {
			list.append(dynamic_cast<const CPyParam*>(vector[i].Ptr())->Param());
		}
		return list;
	}

	py::list GetPopulation() const
	{
		const CArray<CFunctionParamVector>& pop = evolution->Evolution().GetPopulation();
		py::list list;
		for(int i = 0; i < pop.Size(); i++) {
			py::list values;
			for (int j = 0; j < pop[i].Size(); j++) {
				values.append(dynamic_cast<const CPyParam*>(pop[i][j].Ptr())->Param());
			}
			list.append(values);
		}
		return list;
	}

	py::list GetPopulationFuncValues() const
	{
		const CArray<CFunctionParam>& values = evolution->Evolution().GetPopulationFuncValues();
		py::list list;
		for(int i = 0; i < values.Size(); i++) {
			list.append(dynamic_cast<const CPyParam*>(values[i].Ptr())->Param());
		}
		return list;
	}


private:
	CPtr<CPyFunctionEvaluationOwner> function;
	CPtr<CPyDifferentialEvolutionOwner> evolution;
};

//------------------------------------------------------------------------------------------------------------

void InitializeDifferentialEvolution(py::module& m)
{
	py::class_<CPyDifferentialEvolution>(m, "DifferentialEvolution")
		.def(py::init([]( py::object f, py::list lowerBounds, py::list upperBounds, py::list param_traits,
			py::object result_traits, float fluctuation, float cr, int population, int maxGenerationCount, int maxNonGrowingCount )
		{
			CPtr<CPyFunctionEvaluationOwner> functional = new CPyFunctionEvaluationOwner(f, lowerBounds, upperBounds,
				param_traits, result_traits);
			CPtr<CPyDifferentialEvolutionOwner> evolution = new CPyDifferentialEvolutionOwner(functional->Function(),
				fluctuation, cr, population); 
			evolution->Evolution().SetMaxGenerationCount(maxGenerationCount);
			evolution->Evolution().SetMaxNonGrowingBestValue(maxNonGrowingCount);

			return new CPyDifferentialEvolution( functional, evolution );
		}))
		.def("build_next_generation", &CPyDifferentialEvolution::BuildNextGeneration, py::return_value_policy::reference)
		.def("run", &CPyDifferentialEvolution::Run, py::return_value_policy::reference)
		.def("get_optimal_vector", &CPyDifferentialEvolution::GetOptimalVector, py::return_value_policy::reference)
		.def("get_population", &CPyDifferentialEvolution::GetPopulation, py::return_value_policy::reference)
		.def("get_population_function_values", &CPyDifferentialEvolution::GetPopulationFuncValues, py::return_value_policy::reference)
	;
}
