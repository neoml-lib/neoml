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

#include <NeoML/Dnn/Layers/MultichannelLookupLayer.h>

namespace NeoML {

CMultichannelLookupLayer::CMultichannelLookupLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnMultichannelLookupLayer", true ),
	useFrameworkLearning( false )
{
}

void CMultichannelLookupLayer::SetDimensions(const CArray<CLookupDimension>& d)
{
	d.CopyTo(dimensions);
}

const CDnnBlob* CMultichannelLookupLayer::GetEmbeddings(int i) const
{
	NeoAssert( i >= 0 && i < dimensions.Size() );

	if( i < getParams().Size() ) {
		return getParams()[i];
	} else {
		return 0;
	}
}

void CMultichannelLookupLayer::SetEmbeddings( const CPtr<CDnnBlob>& data, int i )
{
	NeoAssert( i >= 0 && i < dimensions.Size() );

	if(getParams().Size() <= i) {
		getParams().SetSize(GetDimensions().Size());
	}
	
	if( data != 0 ) {
		NeoAssert(data->GetObjectCount() == GetDimensions()[i].VectorCount);
		NeoAssert(data->GetObjectSize() == GetDimensions()[i].VectorSize);
		getParams()[i] = data->GetCopy();
	} else {
		getParams()[i] = 0;
	}
}

void CMultichannelLookupLayer::SetEmbeddings( CPtr<CDnnBlob>& data, int i, bool copy )
{
	NeoAssert( i >= 0 && i < dimensions.Size() );

	if(getParams().Size() <= i) {
		getParams().SetSize(GetDimensions().Size());
	}

	if( data != 0 ) {
		NeoAssert(data->GetObjectCount() == GetDimensions()[i].VectorCount);
		NeoAssert(data->GetObjectSize() == GetDimensions()[i].VectorSize);
		if( copy ) {
			getParams()[i] = data->GetCopy();
		} else {
			getParams()[i] = data;
		}
	} else {
		getParams()[i] = 0;
	}
}

void CMultichannelLookupLayer::SetUseFrameworkLearning(bool _useFrameworkLearning)
{
	if(_useFrameworkLearning && !useFrameworkLearning) {
		paramBlobs.SetSize(ownParams.Size());
		for(int i = 0; i < paramBlobs.Size(); ++i) {
			paramBlobs[i] = ownParams[i];
		}
		ForceReshape();
	} else if(!_useFrameworkLearning && useFrameworkLearning) {
		ownParams.SetSize(paramBlobs.Size());
		for(int i = 0; i < ownParams.Size(); ++i) {
			ownParams[i] = paramBlobs[i];
		}
		ForceReshape();
	}
	useFrameworkLearning = _useFrameworkLearning;
}

CArchive& operator << (CArchive& archive, const CLookupDimension& d) {
	return archive << d.VectorCount << d.VectorSize;
}

CArchive& operator >> (CArchive& archive, CLookupDimension& d) {
	return archive >> d.VectorCount >> d.VectorSize;
}

static const int MultichannelLookupLayerVersion = 2000;

void CMultichannelLookupLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MultichannelLookupLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
	
	dimensions.Serialize(archive);
	archive.Serialize(useFrameworkLearning);
	SerializeBlobs( MathEngine(), archive, ownParams );
}

void CMultichannelLookupLayer::Initialize(CDnnInitializer* init)
{
	if(getParams().Size() != GetDimensions().Size()) {
		getParams().SetSize(GetDimensions().Size());
	}

	for(int i = 0; i < getParams().Size(); i++) {
		if(getParams()[i] == 0) {
			getParams()[i] = CDnnBlob::CreateDataBlob(MathEngine(), CT_Float, 1, GetDimensions()[i].VectorCount, GetDimensions()[i].VectorSize);
			if(init != 0) {
				init->InitializeLayerParams(*getParams()[i], 1);
			} else {
				getParams()[i]->Clear();
			}
		}
	}
}

void CMultichannelLookupLayer::Reshape()
{
	// Check the input blob parameters
	CheckInputs();
	CheckArchitecture(inputDescs[0].Channels() >= GetDimensions().Size(),
		GetName(), "MultichannelLookup layer must have input with more channels");

	Initialize(GetDnn()->GetInitializer());
	NeoAssert(getParams().Size() == GetDimensions().Size());

	int outputChannels = inputDescs[0].Channels() - GetDimensions().Size();
	for (int i = 0; i < getParams().Size(); i++) {
		NeoAssert(getParams()[i] != 0);
		NeoAssert(getParams()[i]->GetObjectCount() == GetDimensions()[i].VectorCount);
		NeoAssert(getParams()[i]->GetObjectSize() == GetDimensions()[i].VectorSize);

		outputChannels += GetDimensions()[i].VectorSize;
	}
	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDataType( CT_Float );
	outputDescs[0].SetDimSize( BD_Channels, outputChannels );
}

void CMultichannelLookupLayer::RunOnce()
{
	CArray<CConstFloatHandle> lookupTables;
	for (int i = 0; i < getParams().Size(); i++) {
		lookupTables.Add(getParams()[i]->GetData());
	}
	// Fill in the output blob with the vector representations from the embeddings
	if(inputBlobs[0]->GetDataType() == CT_Float) {
		MathEngine().VectorMultichannelLookupAndCopy(inputBlobs[0]->GetObjectCount() * inputBlobs[0]->GetGeometricalSize(),
			inputBlobs[0]->GetChannelsCount(), inputBlobs[0]->GetData(),
			lookupTables.GetPtr(), GetDimensions().GetPtr(), GetDimensions().Size(),
			outputBlobs[0]->GetData(), outputBlobs[0]->GetChannelsCount());
	} else {
		MathEngine().VectorMultichannelLookupAndCopy(inputBlobs[0]->GetObjectCount() * inputBlobs[0]->GetGeometricalSize(),
			inputBlobs[0]->GetChannelsCount(), inputBlobs[0]->GetData<int>(),
			lookupTables.GetPtr(), GetDimensions().GetPtr(), GetDimensions().Size(),
			outputBlobs[0]->GetData(), outputBlobs[0]->GetChannelsCount());
	}
}

void CMultichannelLookupLayer::BackwardOnce()
{
	// Similar to an input layer, so we don't need to do anything on a backward pass
	NeoAssert(0);
}

void CMultichannelLookupLayer::LearnOnce()
{
	CFloatHandleStackVar learningRate( MathEngine() );

	if(useFrameworkLearning) {
		CArray<CFloatHandle> lookupTables;
		for(int i = 0; i < getParams().Size(); i++) {
			lookupTables.Add(paramDiffBlobs[i]->GetData());
		}

		learningRate.SetValue( 1 );

		if(inputBlobs[0]->GetDataType() == CT_Float) {
			MathEngine().VectorMultichannelLookupAndAddToTable(inputBlobs[0]->GetObjectCount() * inputBlobs[0]->GetGeometricalSize(),
				inputBlobs[0]->GetChannelsCount(), inputBlobs[0]->GetData(),
				lookupTables.GetPtr(), GetDimensions().GetPtr(), GetDimensions().Size(),
				learningRate, outputDiffBlobs[0]->GetData(), outputBlobs[0]->GetChannelsCount());
		} else {
			MathEngine().VectorMultichannelLookupAndAddToTable(inputBlobs[0]->GetObjectCount() * inputBlobs[0]->GetGeometricalSize(),
				inputBlobs[0]->GetChannelsCount(), inputBlobs[0]->GetData<int>(),
				lookupTables.GetPtr(), GetDimensions().GetPtr(), GetDimensions().Size(),
				learningRate, outputDiffBlobs[0]->GetData(), outputBlobs[0]->GetChannelsCount());
		}
	} else {
		CArray<CFloatHandle> lookupTables;
		for(int i = 0; i < getParams().Size(); i++) {
			lookupTables.Add(getParams()[i]->GetData());
		}

		float rate = GetDnn()->GetSolver()->GetLearningRate() * GetBaseLearningRate();
		learningRate.SetValue(-rate);

		// Add to paramDiffs the diffs for the embeddings used when learning
		if(inputBlobs[0]->GetDataType() == CT_Float) {
			MathEngine().VectorMultichannelLookupAndAddToTable(inputBlobs[0]->GetObjectCount() * inputBlobs[0]->GetGeometricalSize(),
				inputBlobs[0]->GetChannelsCount(), inputBlobs[0]->GetData(),
				lookupTables.GetPtr(), GetDimensions().GetPtr(), GetDimensions().Size(),
				learningRate, outputDiffBlobs[0]->GetData(), outputBlobs[0]->GetChannelsCount());
		} else {
			MathEngine().VectorMultichannelLookupAndAddToTable(inputBlobs[0]->GetObjectCount() * inputBlobs[0]->GetGeometricalSize(),
				inputBlobs[0]->GetChannelsCount(), inputBlobs[0]->GetData<int>(),
				lookupTables.GetPtr(), GetDimensions().GetPtr(), GetDimensions().Size(),
				learningRate, outputDiffBlobs[0]->GetData(), outputBlobs[0]->GetChannelsCount());
		}
	}
}

void CMultichannelLookupLayer::Word2VecStep( IMathEngine& mathEngine, int batchSize,
	CMultichannelLookupLayer& word2vecLayer, CMultichannelLookupLayer& context2vecLayer,
	const CConstIntHandle& positiveSampleMatrix, int positiveCount,
	const CConstIntHandle& negativeSampleMatrix, int negativeCount,
	const CConstFloatHandle& positiveWeights, const CConstFloatHandle& negativeWeights,
	const CConstIntHandle& wordMatrix, const CConstFloatHandle& learningRate, const CFloatHandle& loss)
{
	NeoAssert(word2vecLayer.getParams().Size() == 1);
	NeoAssert(word2vecLayer.getParams()[0] != 0);

	int wordCount = word2vecLayer.getParams()[0]->GetObjectCount();
	int vectorLen = word2vecLayer.getParams()[0]->GetObjectSize();
	CFloatHandle word2vec = word2vecLayer.getParams()[0]->GetData();

	NeoAssert(context2vecLayer.getParams().Size() == 1);
	NeoAssert(context2vecLayer.getParams()[0] != 0);
	NeoAssert(wordCount == context2vecLayer.getParams()[0]->GetObjectCount());
	NeoAssert(vectorLen == context2vecLayer.getParams()[0]->GetObjectSize());

	CFloatHandle context2vec = context2vecLayer.getParams()[0]->GetData();

	// Execute
	int totalPositive = batchSize * positiveCount;
	int totalNegative = batchSize * negativeCount;

	CFloatHandleStackVar buffer( mathEngine, totalPositive + totalNegative + batchSize * vectorLen + 1 );
	CFloatHandle positiveRes = buffer.GetHandle();
	CFloatHandle negativeRes = positiveRes + totalPositive;
	CFloatHandle diff = negativeRes + totalNegative;
	CFloatHandle temp = buffer.GetHandle() + buffer.Size() - 1;

	// Calculate correlation between the word vectors and their contexts
	// Should be positive for positive contexts and negative for negative contexts
	mathEngine.MultiplyLookupMatrixByLookupVector(batchSize,
		CLookupMatrix(context2vec, wordCount, vectorLen, positiveSampleMatrix, positiveCount),
		CLookupVector(word2vec, wordCount, vectorLen, wordMatrix), positiveRes, totalPositive);
	mathEngine.MultiplyLookupMatrixByLookupVector(batchSize,
		CLookupMatrix(context2vec, wordCount, vectorLen, negativeSampleMatrix, negativeCount),
		CLookupVector(word2vec, wordCount, vectorLen, wordMatrix), negativeRes, totalNegative);

	// A sigmoid for probabilistic estimate of the correlation 
	// (invert the value for the negative contexts: sig(-x) = 1 - sig(x))
	mathEngine.VectorSigmoid(positiveRes, positiveRes, totalPositive + totalNegative);

	if(!loss.IsNull()) {
		CFloatHandleStackVar lossBuffer( mathEngine, totalPositive + totalNegative );
		CFloatHandle positiveTemp = lossBuffer.GetHandle();
		CFloatHandle negativeTemp = positiveTemp + totalPositive;

		// Calculate loss = -ln(sig(<word, context>)) for positive contexts
		mathEngine.VectorNegLog(positiveRes, positiveTemp, totalPositive);
		// Calculate loss = -ln(sig(-<word, context>)) = ln(1 - sig(<word, context>)) for negative contexts
		mathEngine.VectorFill(negativeTemp, 1, totalNegative);
		mathEngine.VectorSub(negativeTemp, negativeRes, negativeTemp, totalNegative);
		mathEngine.VectorNegLog(negativeTemp, negativeTemp, totalNegative);

		if(!positiveWeights.IsNull()) {
			mathEngine.VectorEltwiseMultiply(positiveTemp, positiveWeights, positiveTemp, totalPositive);
		}
		if(!negativeWeights.IsNull()) {
			mathEngine.VectorEltwiseMultiply(negativeTemp, negativeWeights, negativeTemp, totalNegative);
		}

		mathEngine.VectorSum(positiveTemp, totalPositive + totalNegative, loss);
		temp.SetValue( 1.f / (totalPositive + totalNegative) );
		mathEngine.VectorEltwiseMultiply(loss, temp, loss, 1);
	}

	// Calculate loss gradient
	temp.SetValue(-1.f);
	mathEngine.VectorAddValue(positiveRes, positiveRes, totalPositive, temp);

	// Take weights into account
	if(!positiveWeights.IsNull()) {
		mathEngine.VectorEltwiseMultiply(positiveRes, positiveWeights, positiveRes, totalPositive);
	}
	if(!negativeWeights.IsNull()) {
		mathEngine.VectorEltwiseMultiply(negativeRes, negativeWeights, negativeRes, totalNegative);
	}

	// Take into account learning rate and batchSize
	mathEngine.VectorNegMultiply(positiveRes, positiveRes, totalPositive + totalNegative, learningRate);

	// Calculate diff for the word2vec matrix
	mathEngine.MultiplyTransposedLookupMatrixByVector(batchSize,
		CLookupMatrix(context2vec, wordCount, vectorLen, positiveSampleMatrix, positiveCount),
		positiveRes, diff, batchSize * vectorLen);
	mathEngine.MultiplyTransposedLookupMatrixByVectorAndAdd(batchSize,
		CLookupMatrix(context2vec, wordCount, vectorLen, negativeSampleMatrix, negativeCount),
		positiveRes, diff, batchSize * vectorLen);

	// Calculate diff for the context2vec matrix and apply it directly to context2vec
	mathEngine.MultiplyVectorByTransposedLookupVectorAndAddToTable(batchSize,
		context2vec, wordCount, vectorLen, positiveSampleMatrix,
		positiveRes, positiveCount, CLookupVector(word2vec, wordCount, vectorLen, wordMatrix));
	mathEngine.MultiplyVectorByTransposedLookupVectorAndAddToTable(batchSize,
		context2vec, wordCount, vectorLen, negativeSampleMatrix,
		negativeRes, negativeCount, CLookupVector(word2vec, wordCount, vectorLen, wordMatrix));

	// Apply diff to the word2vec matrix
	mathEngine.MatrixSpreadRowsAdd(diff, batchSize, vectorLen, word2vec, wordCount, wordMatrix);
}

}
