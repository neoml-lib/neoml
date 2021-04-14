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

#include "PyTrainingModel.h"
#include "PyMemoryFile.h"

class CPyMemoryProblem : public IProblem {
public:
	CPyMemoryProblem( int height, int width, const int* columns, const float* values, const int* rowPtr,
		const int* _classes, const float* _weights ) :
		classCount( 0 ),
		classes( _classes ),
		weights( _weights )
	{
		desc.Height = height;
		desc.Width = width;
		desc.Columns = const_cast<int*>( columns );
		desc.Values = const_cast<float*>( values );
		desc.PointerB = const_cast<int*>( rowPtr );
		desc.PointerE = const_cast<int*>( rowPtr ) + 1;
			
		for( int i = 0; i < height; i++ ) {
			if( classCount < classes[i] ) {
				classCount = classes[i];
			}
		}
		classCount++;
	}

	// IProblem interface methods:
	virtual int GetClassCount() const { return classCount; }
	virtual int GetFeatureCount() const { return desc.Width; }
	virtual bool IsDiscreteFeature( int index ) const { return false; }
	virtual int GetVectorCount() const { return desc.Height; }
	virtual int GetClass( int index ) const { return classes[index]; }
	virtual CFloatMatrixDesc GetMatrix() const { return desc; }
	virtual double GetVectorWeight( int index ) const { return weights[index]; };

private:
	CFloatMatrixDesc desc;
	int classCount;
	const int* classes;
	const float* weights;
};

//------------------------------------------------------------------------------------------------------------

class CPyMemoryRegressionProblem : public IRegressionProblem {
public:
	CPyMemoryRegressionProblem( int height, int width, const int* columns, const float* values, const int* rowPtr, const float* _values, const float* _weights ) :
		values( _values ),
		weights( _weights )
	{
		desc.Height = height;
		desc.Width = width;
		desc.Columns = const_cast<int*>( columns );
		desc.Values = const_cast<float*>( values );
		desc.PointerB = const_cast<int*>( rowPtr );
		desc.PointerE = const_cast<int*>( rowPtr ) + 1;
	}

	// IRegressionProblem interface methods:
	virtual int GetFeatureCount() const { return desc.Width; }
	virtual int GetVectorCount() const { return desc.Height; }
	virtual CFloatMatrixDesc GetMatrix() const { return desc; }
	virtual double GetVectorWeight( int index ) const { return weights[index]; };
	virtual double GetValue( int index ) const { return values[index]; }

private:
	CFloatMatrixDesc desc;
	const float* values;
	const float* weights;
};

//------------------------------------------------------------------------------------------------------------

class CPyModel {
public:
	CPyModel() {}
	CPyModel( IModel* model ) : ptr( model ) {}
	CPyModel( const std::string& path );

	IModel* GetModel() const { return ptr; }

	void Store( const std::string& path );

	py::array_t<double> Classify( py::array indices, py::array data, py::array row, bool isSparse );

private:
	CPtr<IModel> ptr;
};

CPyModel::CPyModel( const std::string& path )
{
	CArchiveFile file( path.c_str(), CArchive::load );
	CArchive archive( &file, CArchive::load );
	SerializeModel( archive, ptr );
}

void CPyModel::Store( const std::string& path )
{
	CArchiveFile file( path.c_str(), CArchive::store );
	CArchive archive( &file, CArchive::store );
	SerializeModel( archive, ptr );
}

py::array_t<double> CPyModel::Classify( py::array indices, py::array data, py::array row, bool isSparse )
{
	const int* indicesPtr = reinterpret_cast<const int*>( isSparse ? indices.data() : nullptr );
	const float* dataPtr = reinterpret_cast<const float*>( data.data() );
	const int* rowPtr = reinterpret_cast<const int*>( row.data() );

	int classesCount = ptr->GetClassCount();
	int rowCount = static_cast<int>( row.size() ) - 1;

	py::array_t<double, py::array::c_style> totalResult( { rowCount, classesCount } );
	auto r = totalResult.mutable_unchecked<2>();
	for( int i = 0; i < rowCount; i++ ) {
		CFloatVectorDesc vector;
		vector.Size = rowPtr[i+1] - rowPtr[i];
		vector.Values = const_cast<float*>(dataPtr) + rowPtr[i];
		if ( indicesPtr != nullptr ) {
			vector.Indexes = const_cast<int*>(indicesPtr) + rowPtr[i];
		}

		CClassificationResult result;
		ptr->Classify( vector, result );
		for( int j = 0; j < classesCount; j++ ) {
			r(i, j) = result.Probabilities[j].GetValue();
		}
	}

	return totalResult;
}

//------------------------------------------------------------------------------------------------------------

// A regression model wrapper
class CPyRegressionModel {
public:
	CPyRegressionModel() {}
	CPyRegressionModel( IRegressionModel* model ) : ptr( model ) {}
	CPyRegressionModel( const std::string& path );

	IRegressionModel* GetModel() const { return ptr; }

	void Store( const std::string& path );

	py::array_t<double> Predict( py::array indices, py::array data, py::array row, bool isSparse );

private:
	CPtr<IRegressionModel> ptr;
};

CPyRegressionModel::CPyRegressionModel( const std::string& path )
{
	CArchiveFile file( path.c_str(), CArchive::load );
	CArchive archive( &file, CArchive::load );
	SerializeModel( archive, ptr );
}

void CPyRegressionModel::Store( const std::string& path )
{
	CArchiveFile file( path.c_str(), CArchive::store );
	CArchive archive( &file, CArchive::store );
	SerializeModel( archive, ptr );
}

py::array_t<double> CPyRegressionModel::Predict( py::array indices, py::array data, py::array row, bool isSparse )
{
	const int* indicesPtr = reinterpret_cast<const int*>( isSparse ? indices.data() : nullptr );
	const float* dataPtr = reinterpret_cast<const float*>( data.data() );
	const int* rowPtr = reinterpret_cast<const int*>( row.data() );

	int rowCount = static_cast<int>( row.size() ) - 1;

	py::array_t<double, py::array::c_style> totalResult( rowCount );
	auto r = totalResult.mutable_unchecked<1>();
	for( int i = 0; i < rowCount; i++ ) {
		CFloatVectorDesc vector;
		vector.Size = rowPtr[i+1] - rowPtr[i];
		vector.Values = const_cast<float*>(dataPtr) + rowPtr[i];
		if ( indicesPtr != nullptr ) {
			vector.Indexes = const_cast<int*>(indicesPtr) + rowPtr[i];
		}

		r(i) = ptr->Predict( vector );
	}

	return totalResult;
}

//------------------------------------------------------------------------------------------------------------

class CPyTrainingModelOwner : public IObject {
public:
	explicit CPyTrainingModelOwner( ITrainingModel* _classifier ) : classifier( _classifier ) {}
	virtual ~CPyTrainingModelOwner() { delete classifier; }

	ITrainingModel& TrainingModel() { return *classifier; }

private:
	ITrainingModel* classifier;
};

class CPyTrainingModel {
public:
	explicit CPyTrainingModel( ITrainingModel* classifier ) : owner( new CPyTrainingModelOwner( classifier ) ) {}
	explicit CPyTrainingModel( CPyTrainingModelOwner* _owner ) : owner( _owner ) {}
	virtual ~CPyTrainingModel() {}

	CPyModel TrainClassifier( py::array indices, py::array data, py::array rowPtr, bool isSparse, int featureCount, py::array classes, py::array weight );

	CPyRegressionModel TrainRegressor( py::array indices, py::array data, py::array rowPtr, bool isSparse, int featureCount, py::array values, py::array weight );

	CPyTrainingModelOwner* GetOwner() const { return owner; }
private:
	CPtr<CPyTrainingModelOwner> owner;
};

CPyModel CPyTrainingModel::TrainClassifier( py::array indices, py::array data, py::array rowPtr, bool isSparse, int featureCount, py::array classes, py::array weight )
{
	CPtr<CPyMemoryProblem> problem = new CPyMemoryProblem( static_cast<int>( classes.size() ), featureCount,
		reinterpret_cast<const int*>( isSparse ? indices.data() : nullptr ), reinterpret_cast<const float*>( data.data() ),
		reinterpret_cast<const int*>( rowPtr.data() ), reinterpret_cast<const int*>( classes.data() ),
		reinterpret_cast<const float*>( weight.data() ) );
	CPtr<IModel> model = owner->TrainingModel().Train( *(problem.Ptr()) );

	return CPyModel( model.Ptr() );
}

CPyRegressionModel CPyTrainingModel::TrainRegressor( py::array indices, py::array data, py::array rowPtr, bool isSparse, int featureCount, py::array values, py::array weight )
{
	CPtr<CPyMemoryRegressionProblem> problem = new CPyMemoryRegressionProblem( static_cast<int>( values.size() ), featureCount,
		reinterpret_cast<const int*>( isSparse ? indices.data() : nullptr ), reinterpret_cast<const float*>( data.data() ),
		reinterpret_cast<const int*>( rowPtr.data() ), reinterpret_cast<const float*>( values.data() ),
		reinterpret_cast<const float*>( weight.data() ) );
	CPtr<IRegressionModel> model = dynamic_cast<IRegressionTrainingModel&>(owner->TrainingModel()).TrainRegression( *(problem.Ptr()) );

	return CPyRegressionModel( model.Ptr() );
}

//------------------------------------------------------------------------------------------------------------

class CPyDecisionTree : public CPyTrainingModel {
public:
	explicit CPyDecisionTree( const CDecisionTreeTrainingModel::CParams& p ) : CPyTrainingModel( new CDecisionTreeTrainingModel(p) ) {}
};

//------------------------------------------------------------------------------------------------------------

class CPySvm : public CPyTrainingModel {
public:
	explicit CPySvm( const CSvmBinaryClassifierBuilder::CParams& p ) : CPyTrainingModel( new CSvmBinaryClassifierBuilder(p) ) {}
};

//------------------------------------------------------------------------------------------------------------

class CPyLinear : public CPyTrainingModel {
public:
	explicit CPyLinear( const CLinearBinaryClassifierBuilder::CParams& p ) : CPyTrainingModel( new CLinearBinaryClassifierBuilder(p) ) {}
};

//------------------------------------------------------------------------------------------------------------

class CPyGradientBoost : public CPyTrainingModel {
public:
	CPyGradientBoost( const CGradientBoost::CParams& p, CRandom* randomPtr ) : CPyTrainingModel( new CGradientBoost(p) ), random( randomPtr ) {}
	~CPyGradientBoost() { if( random != 0 ) { delete random; } }
private:
	CRandom* random;
};

//------------------------------------------------------------------------------------------------------------

void InitializeTrainingModel(py::module& m)
{
	py::class_<CPyModel>(m, "Model")
		.def(py::init([]( const std::string& path ) { return new CPyModel( path ); } ) )
		.def("classify", &CPyModel::Classify, py::return_value_policy::reference)
		.def("store", &CPyModel::Store, py::return_value_policy::reference)
		.def(py::pickle(
			[](const CPyModel& pyModel) {
				CPyMemoryFile file;
				CArchive archive( &file, CArchive::store );
				CPtr<IModel> model( pyModel.GetModel() );
				SerializeModel( archive, model );
				archive.Close();
				file.Close();
				return py::make_tuple( file.GetBuffer() );
			},
			[](py::tuple t) {
				if( t.size() != 1 ) {
					throw std::runtime_error("Invalid state!");
				}

				auto t0_array = t[0].cast<py::array>();
				CPyMemoryFile file( t0_array );
				CArchive archive( &file, CArchive::load );
				CPtr<IModel> model;
				SerializeModel( archive, model );
				return new CPyModel( model.Ptr() );
			}
		))
	;
 
//------------------------------------------------------------------------------------------------------------

	py::class_<CPyRegressionModel>(m, "RegressionModel")
		.def(py::init([]( const std::string& path ) { return new CPyRegressionModel( path ); } ) )
		.def("predict", &CPyRegressionModel::Predict, py::return_value_policy::reference)
		.def("store", &CPyRegressionModel::Store, py::return_value_policy::reference)
		.def(py::pickle(
			[](const CPyRegressionModel& pyModel) {
				CPyMemoryFile file;
				CArchive archive( &file, CArchive::store );
				CPtr<IRegressionModel> model( pyModel.GetModel() );
				SerializeModel( archive, model );
				archive.Close();
				file.Close();
				return py::make_tuple( file.GetBuffer() );
			},
			[](py::tuple t) {
				if( t.size() != 1 ) {
					throw std::runtime_error("Invalid state!");
				}

				auto t0_array = t[0].cast<py::array>();
				CPyMemoryFile file( t0_array );
				CArchive archive( &file, CArchive::load );
				CPtr<IRegressionModel> model;
				SerializeModel( archive, model );
				return new CPyRegressionModel( model.Ptr() );
			}
		))
	;
 
//------------------------------------------------------------------------------------------------------------

	py::class_<CPyTrainingModel>(m, "TrainingModel")
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyDecisionTree, CPyTrainingModel>(m, "DecisionTree")
		.def(
			py::init([]( int min_subset_size, float min_subset_part, int min_split_size, int max_tree_depth, int max_node_count, const std::string& criterion,
						float const_threshold, int random_selected_feature_count )
						{
							CDecisionTreeTrainingModel::CParams p;
							p.SplitCriterion = CDecisionTreeTrainingModel::SC_Count;
							if( criterion == "gini" ) {
								p.SplitCriterion = CDecisionTreeTrainingModel::SC_GiniImpurity;
							} else if( criterion == "information_gain" ) {
								p.SplitCriterion = CDecisionTreeTrainingModel::SC_InformationGain;
							}

							p.MinContinuousSubsetSize = min_subset_size;
							p.MinContinuousSubsetPart = min_subset_part;
							p.MinSplitSize = min_split_size;
							p.MaxTreeDepth = max_tree_depth;
							p.MaxNodesCount = max_node_count;
							p.ConstNodeThreshold = const_threshold;
							p.RandomSelectedFeaturesCount = random_selected_feature_count;

							return new CPyDecisionTree( p );
						})
		)

		.def( "train_classifier", &CPyDecisionTree::TrainClassifier, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPySvm, CPyTrainingModel>(m, "Svm")
		.def( py::init(
			[]( const std::string& kernel, float error_weight, int max_iteration_count, int degree, float gamma, float coeff0,
					float tolerance, int thread_count ) {
				CSvmBinaryClassifierBuilder::CParams p( CSvmKernel::KT_Undefined );
				if( kernel == "linear" ) {
					p.KernelType = CSvmKernel::KT_Linear;
				} else if( kernel == "poly" ) {
					p.KernelType = CSvmKernel::KT_Poly;
				} else if( kernel == "rbf" ) {
					p.KernelType = CSvmKernel::KT_RBF;
				} else if( kernel == "sigmoid" ) {
					p.KernelType = CSvmKernel::KT_Sigmoid;
				}
				p.ErrorWeight = error_weight;
				p.MaxIterations = max_iteration_count;
				p.Degree = degree;
				p.Gamma = gamma;
				p.Coeff0 = coeff0;
				p.Tolerance = tolerance;
				p.ThreadCount = thread_count;

				return new CPySvm( p );
			})
		)

		.def( "train_classifier", &CPySvm::TrainClassifier, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyLinear, CPyTrainingModel>(m, "Linear")
		.def( py::init(
			[]( const std::string& loss, int max_iteration_count, float error_weight, float sigmoid_a, float sigmoid_b,
					float tolerance, bool normalize_error, float l1_reg, int thread_count ) {
				CLinearBinaryClassifierBuilder::CParams p( EF_Count );
				if( loss == "smoothed_hinge" ) {
					p.Function = EF_SmoothedHinge;
				} else if( loss == "binomial" ) {
					p.Function = EF_LogReg;
				} else if( loss == "squared_hinge" ) {
					p.Function = EF_SquaredHinge;
				} else if( loss == "l2" ) {
					p.Function = EF_L2_Regression;
				}
				p.MaxIterations = max_iteration_count;
				p.ErrorWeight = error_weight;
				p.SigmoidCoefficients.A = sigmoid_a;
				p.SigmoidCoefficients.B = sigmoid_b;
				p.Tolerance = tolerance;
				p.NormalizeError = normalize_error;
				p.L1Coeff = l1_reg;
				p.ThreadCount = thread_count;

				return new CPyLinear( p );
			})
		)

		.def( "train_classifier", &CPyLinear::TrainClassifier, py::return_value_policy::reference )
		.def( "train_regressor", &CPyLinear::TrainRegressor, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyGradientBoost, CPyTrainingModel>(m, "GradientBoost")
		.def( py::init(
			[]( const std::string& loss, int iteration_count, float learning_rate, float subsample, float subfeature,
					int random_seed, int max_depth, int max_node_count, float l1_reg, float l2_reg, float prune, int thread_count,
					const std::string& builder_type, int max_bins, float min_subtree_weight ) {
				CGradientBoost::CParams p;
				p.LossFunction = CGradientBoost::LF_Undefined;
				if( loss == "exponential" ) {
					p.LossFunction = CGradientBoost::LF_Exponential;
				} else if( loss == "binomial" ) {
					p.LossFunction = CGradientBoost::LF_Binomial;
				} else if( loss == "squared_hinge" ) {
					p.LossFunction = CGradientBoost::LF_SquaredHinge;
				} else if( loss == "l2" ) {
					p.LossFunction = CGradientBoost::LF_L2;
				}
				p.IterationsCount = iteration_count;
				p.LearningRate = learning_rate;
				p.Subsample = subsample;
				p.Subfeature = subfeature;
				p.Random = new CRandom( random_seed );
				p.MaxTreeDepth = max_depth;
				p.MaxNodesCount = max_node_count;
				p.L1RegFactor = l1_reg;
				p.L2RegFactor = l2_reg;
				p.PruneCriterionValue = prune;
				p.ThreadCount = thread_count;
				p.TreeBuilder = GBTB_Count;
				if( builder_type == "full" ) {
					p.TreeBuilder = GBTB_Full;
				} else if( builder_type == "hist" ) {
					p.TreeBuilder = GBTB_FastHist;
				} else if ( builder_type == "multi_full" ) {
					p.TreeBuilder = GBTB_MultiFull;
				}
				p.MaxBins = max_bins;
				p.MinSubsetWeight = min_subtree_weight;
				p.Representation = GBMR_Linked;

				return new CPyGradientBoost( p, p.Random );
			})
		)

		.def( "train_classifier", &CPyGradientBoost::TrainClassifier, py::return_value_policy::reference )
		.def( "train_regressor", &CPyGradientBoost::TrainRegressor, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	m.def("_cross_validation_score", []( const CPyTrainingModel& classifier, py::array indices, py::array data, py::array rowPtr, 
		bool isSparse, int featureCount, py::array classes, py::array weight, const std::string& scoreName, int parts, bool stratified )
	{
		CPtr<CPyMemoryProblem> problem = new CPyMemoryProblem( static_cast<int>( classes.size() ), featureCount,
			reinterpret_cast<const int*>( isSparse ? indices.data() : nullptr ), reinterpret_cast<const float*>( data.data() ),
			reinterpret_cast<const int*>( rowPtr.data() ), reinterpret_cast<const int*>( classes.data() ),
			reinterpret_cast<const float*>( weight.data() ) );

		CCrossValidationResult results;
		CCrossValidation crossValidation(classifier.GetOwner()->TrainingModel(), problem);
		TScore score = scoreName == "f1" ? F1Score : AccuracyScore;
		crossValidation.Execute( parts, score, results, stratified );

		py::array_t<double, py::array::c_style> scores( results.Success.Size() );
		auto tempScores = scores.mutable_unchecked<1>();
		for( int i = 0; i < results.Success.Size(); i++ ) {
			tempScores(i) = results.Success[i];
		}
		return scores;
	});

}
