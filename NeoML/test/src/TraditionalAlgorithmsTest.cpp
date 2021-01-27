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

#include <TestFixture.h>

#include <random>
#include <limits>

using namespace NeoML;
using namespace NeoMLTest;

class CSvmClassificationTest : public CNeoMLTestFixture {
public:
	static bool InitTestFixture(); // Инициализация теста
	static void DeinitTestFixture(); // Деинициализация теста
};

bool CSvmClassificationTest::InitTestFixture()
{
	return true;
}

void CSvmClassificationTest::DeinitTestFixture()
{
}

//------------------------------------------------------------------------------------------------
class CDenseMemoryProblem : public IProblem {
public:
	CDenseMemoryProblem( int height, int width, float* values, const int* _classes, const float* _weights ) :
		classCount( 0 ),
		classes( _classes ),
		weights( _weights )
	{
		desc.Height = height;
		desc.Width = width;
		desc.Values = values;

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
	virtual bool IsDiscreteFeature( int ) const { return false; }
	virtual int GetVectorCount() const { return desc.Height; }
	virtual int GetClass( int index ) const { return classes[index]; }
	virtual CFloatMatrixDesc GetMatrix() const { return desc; }
	virtual double GetVectorWeight( int index ) const { return weights[index]; };

	static CPtr<CDenseMemoryProblem> Random( int samples, int features, int classes )
	{
		CPtr<CDenseMemoryProblem> res = new CDenseMemoryProblem();

		std::random_device rd;
		std::mt19937 gen( rd() );
		std::uniform_real_distribution<float> df( std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
		std::uniform_int_distribution<int> di( 0, classes - 1 );
		res->valuesArr.SetBufferSize( samples * features );
		res->classesArr.SetBufferSize( samples );
		for( int i = 0; i < samples; ++i ) {
			for( int j = 0; j < features; ++j ) {
				res->valuesArr.Add( df( gen ) );
			}
			res->classesArr.Add( di( gen ) );
		}
		// set weights to 1
		res->weightsArr.Add( 1., samples );
		res->classCount = classes;
		res->classes = res->classesArr.GetPtr();
		res->weights = res->weightsArr.GetPtr();
		res->desc.Height = samples;
		res->desc.Width = features;
		res->desc.Values = res->valuesArr.GetPtr();

		return res;
	}

protected:
	~CDenseMemoryProblem() override = default;

private:
	CDenseMemoryProblem() = default;

	CFloatMatrixDesc desc;
	int classCount;
	const int* classes;
	const float* weights;

	// memory holders when applicable
	CArray<float> valuesArr;
	CArray<int> classesArr;
	CArray<float> weightsArr;
};

CPtr<CDenseMemoryProblem> DenseBinaryProblem = CDenseMemoryProblem::Random( 4000, 20, 2 );

//------------------------------------------------------------------------------------------------

TEST_F( CSvmClassificationTest, RandomDense4000x20 )
{
	CSvmBinaryClassifierBuilder::CParams params( CSvmKernel::KT_Linear );
	CSvmBinaryClassifierBuilder svm( params );
	auto model = svm.Train( *DenseBinaryProblem );
}
