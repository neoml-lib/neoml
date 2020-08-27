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

#include <NeoML/Dnn/Layers/ObjectNormalizationLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CObjectNormalizationLayer::CObjectNormalizationLayer( IMathEngine& mathEngine ) :
	CBaseInPlaceLayer( mathEngine, "CObjectNormalizationLayer", true ),
	epsilon( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	invObjectSize( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) )
{
	paramBlobs.SetSize( PN_Count );
	SetEpsilon( 1e-5f );
}

static const int ObjectNormalizationLayerVersion = 2000;

void CObjectNormalizationLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ObjectNormalizationLayerVersion );
	CBaseInPlaceLayer::Serialize( archive );
	
	float epsilonValue = archive.IsStoring() ? GetEpsilon() : 0.f;
	
	archive.Serialize( epsilonValue );
	
	if( archive.IsLoading() ) {
		SetEpsilon( epsilonValue );
	}
}

void CObjectNormalizationLayer::SetEpsilon( float newEpsilon )
{
	NeoAssert( newEpsilon > 0 );
	epsilon->GetData().SetValue( newEpsilon );
}

float CObjectNormalizationLayer::GetEpsilon() const
{
	return epsilon->GetData().GetValue();
}

void CObjectNormalizationLayer::SetScale( const CPtr<CDnnBlob>& newScale )
{
	if( newScale == nullptr ) {
		NeoAssert( Scale() == nullptr || GetDnn() == nullptr );
		Scale() = nullptr;
	} else if( Scale() != nullptr && GetDnn() != nullptr ) {
		NeoAssert( Scale()->GetDataSize() == newScale->GetDataSize() );
		Scale()->CopyFrom( newScale );
	} else {
		Scale() = newScale->GetCopy();
	}
}

CPtr<CDnnBlob> CObjectNormalizationLayer::GetScale() const
{
	if( Scale() == nullptr ) {
		return nullptr;
	}
	return Scale()->GetCopy();
}

void CObjectNormalizationLayer::SetBias( const CPtr<CDnnBlob>& newBias )
{
	if( newBias == nullptr ) {
		NeoAssert( Bias() == nullptr || GetDnn() == nullptr );
		Bias() = nullptr;
	} else if( Bias() != nullptr && GetDnn() != nullptr ) {
		NeoAssert( Bias()->GetDataSize() == newBias->GetDataSize() );
		Bias()->CopyFrom( newBias );
	} else {
		Bias() = newBias->GetCopy();
	}
}

CPtr<CDnnBlob> CObjectNormalizationLayer::GetBias() const
{
	if( Bias() == nullptr ) {
		return nullptr;
	}
	return Bias()->GetCopy();
}

void CObjectNormalizationLayer::OnReshaped()
{
	CheckArchitecture( GetInputCount() == 1, GetName(), "layer must have exactly 1 input" );
	CheckArchitecture( GetOutputCount() == 1, GetName(), "Source layer has more than 1 output" );

	CBlobDesc paramDesc;
	paramDesc.SetDimSize( BD_Channels, inputDescs[0].ObjectSize() );
	if( Scale() == nullptr || Scale()->GetDataSize() != paramDesc.BlobSize() ) {
		Scale() = CDnnBlob::CreateBlob( MathEngine(), CT_Float, paramDesc );
		Scale()->Fill( 1.f );
	}
	if( Bias() == nullptr || Bias()->GetDataSize() != paramDesc.BlobSize() ) {
		Bias() = CDnnBlob::CreateBlob( MathEngine(), CT_Float, paramDesc );
		Bias()->Clear();
	}

	normalizedInput = nullptr;
	if( IsBackwardPerformed() || IsLearningPerformed() ) {
		// This value is used in both LearnOnce and BackwardOnce.
		normalizedInput = CDnnBlob::CreateBlob( MathEngine(), CT_Float, inputDescs[0] );

		if( GetDnn()->IsRecurrentMode() ) {
			RegisterRuntimeBlob( normalizedInput );
		}
	}

	internalParams = nullptr;
	if( IsBackwardPerformed() ) {
		// This value is used only in BackwardOnce.
		CBlobDesc internalParamDesc;
		internalParamDesc.SetDimSize( BD_BatchWidth, IPN_Count );

		if( IsBackwardPerformed() && GetDnn()->IsRecurrentMode() ) {
			internalParamDesc.SetDimSize( BD_Channels, inputDescs[0].BatchWidth() * inputDescs[0].ListSize() );
			internalParamDesc.SetDimSize( BD_BatchLength, inputDescs[0].BatchLength() );
		} else {
			internalParamDesc.SetDimSize( BD_Channels, inputDescs[0].ObjectCount() );
		}

		internalParams = CDnnBlob::CreateBlob( MathEngine(), CT_Float, internalParamDesc );

		if( GetDnn()->IsRecurrentMode() ) {
			RegisterRuntimeBlob( internalParams );
		}
	}

	outputDiffBackup = nullptr;
	if( IsBackwardPerformed() && IsLearningPerformed() ) {
		// This layer is in-place.
		// That means, it's possible that inputDiffBlobs[0] == outputDiffBlobs[0].
		// But both BackwardOnce and LearnOnce are using outputDiffBlobs[0].
		// In that case we need to store values of outputDiffBlobs[0].
		outputDiffBackup = ( IsBackwardPerformed() && IsLearningPerformed() )
			? CDnnBlob::CreateBlob( MathEngine(), CT_Float, inputDescs[0] ) : nullptr;

		if( GetDnn()->IsRecurrentMode() ) {
			RegisterRuntimeBlob( outputDiffBackup );
		}
	}

	invObjectSize->GetData().SetValue( 1.f / inputDescs[0].ObjectSize() );

	inputDescs.CopyTo( outputDescs );
}

void CObjectNormalizationLayer::RunOnce()
{
	const int objectCount = inputBlobs[0]->GetObjectCount();

	if( internalParams == nullptr ) {
		CFloatHandleStackVar meanAndVarBuff( MathEngine(), 2 * objectCount );
		runOnceImpl( meanAndVarBuff.GetHandle(), meanAndVarBuff.GetHandle() + objectCount,
			normalizedInput == nullptr ? outputBlobs[0]->GetData() : normalizedInput->GetData() );
	} else {
		runOnceImpl( internalParams->GetObjectData( IPN_NegMean ), internalParams->GetObjectData( IPN_InvSqrtVariance ),
			normalizedInput == nullptr ? outputBlobs[0]->GetData() : normalizedInput->GetData() );
	}
}

void CObjectNormalizationLayer::runOnceImpl( const CFloatHandle& negMean, const CFloatHandle& invSqrtVar,
	const CFloatHandle& inputNorm )
{
	calcMean( negMean );
	calcVar( negMean, invSqrtVar );
	normalizeInput( negMean, invSqrtVar, inputNorm );
	applyScaleAndBias( inputNorm );
}

void CObjectNormalizationLayer::calcMean( const CFloatHandle& negMean )
{
	MathEngine().SumMatrixColumns( negMean, inputBlobs[0]->GetData(),
		inputBlobs[0]->GetObjectCount(), inputBlobs[0]->GetObjectSize() );
	MathEngine().VectorNegMultiply( negMean, negMean, inputBlobs[0]->GetObjectCount(),
		invObjectSize->GetData() );
}

void CObjectNormalizationLayer::calcVar( const CConstFloatHandle& negMean, const CFloatHandle& invSqrtVar )
{
	const int objectCount = inputBlobs[0]->GetObjectCount();
	const int objectSize = inputBlobs[0]->GetObjectSize();

	CConstFloatHandle input = inputBlobs[0]->GetData();

	CFloatHandleStackVar temp( MathEngine(), inputBlobs[0]->GetDataSize() );

	MathEngine().AddVectorToMatrixColumns( input, temp, objectCount, objectSize, negMean );
	MathEngine().VectorEltwiseMultiply( temp, temp, temp, temp.Size() );
	MathEngine().SumMatrixColumns( invSqrtVar, temp, objectCount, objectSize );

	// Normalize the variance and calculate the inverse to the standard deviation.
	MathEngine().VectorMultiply( invSqrtVar, invSqrtVar, objectCount, invObjectSize->GetData() );
	MathEngine().VectorAddValue( invSqrtVar, invSqrtVar, objectCount, epsilon->GetData() );
	MathEngine().VectorSqrt( invSqrtVar, invSqrtVar, objectCount );
	MathEngine().VectorInv( invSqrtVar, invSqrtVar, objectCount );
}

void CObjectNormalizationLayer::normalizeInput( const CConstFloatHandle& negMean, const CConstFloatHandle& invSqrtVar,
	const CFloatHandle& inputNorm )
{
	const int objectCount = inputBlobs[0]->GetObjectCount();
	const int objectSize = inputBlobs[0]->GetObjectSize();

	CConstFloatHandle input = inputBlobs[0]->GetData();
	const int outSize = normalizedInput == nullptr ? outputBlobs[0]->GetDataSize() : normalizedInput->GetDataSize();

	MathEngine().AddVectorToMatrixColumns( input, inputNorm, objectCount, objectSize, negMean );
	MathEngine().MultiplyDiagMatrixByMatrix( invSqrtVar, objectCount, inputNorm, objectSize, inputNorm, outSize );
}

void CObjectNormalizationLayer::applyScaleAndBias( const CConstFloatHandle& inputNorm )
{
	const int objectCount = inputBlobs[0]->GetObjectCount();
	const int objectSize = inputBlobs[0]->GetObjectSize();

	CFloatHandle output = outputBlobs[0]->GetData();
	CConstFloatHandle scale = Scale()->GetData();
	CConstFloatHandle bias = Bias()->GetData();

	MathEngine().MultiplyMatrixByDiagMatrix( inputNorm, objectCount, objectSize, scale, output, outputBlobs[0]->GetDataSize() );
	MathEngine().AddVectorToMatrixRows( 1, output, output, objectCount, objectSize, bias );
}

void CObjectNormalizationLayer::BackwardOnce()
{
	const int objectCount = inputBlobs[0]->GetObjectCount();
	const int objectSize = inputBlobs[0]->GetObjectSize();
	const int dataSize = objectCount * objectSize;

	CConstFloatHandle input = normalizedInput->GetData();
	CFloatHandle inputDiff = inputDiffBlobs[0]->GetData();
	CConstFloatHandle outputDiff = outputDiffBlobs[0]->GetData();
	CConstFloatHandle scale = Scale()->GetData();
	CConstFloatHandle invSqrtVar = internalParams->GetObjectData( IPN_InvSqrtVariance );

	if( outputDiffBackup != nullptr ) {
		MathEngine().VectorCopy( outputDiffBackup->GetData(), outputDiff, outputDiffBackup->GetDataSize() );
	}

	// Average is used multiple times in RunOnce.
	// But it is used neither in BackwardOnce nor in LearnOnce.
	// That's why it's possible to reuse it here as a buffer.
	CFloatHandle inputMultiplier = internalParams->GetObjectData( IPN_NegMean );
	
	// Buffer for next operations.
	CFloatHandleStackVar buff( MathEngine(), dataSize );

	{
		CFloatHandle outDiffMultipliedByInput = buff.GetHandle();
		MathEngine().VectorEltwiseMultiply( outputDiff, input, outDiffMultipliedByInput, dataSize );
		MathEngine().MultiplyMatrixByMatrix( 1, outDiffMultipliedByInput, objectCount, objectSize, scale, 1, inputMultiplier, internalParams->GetObjectSize() );
		MathEngine().VectorNegMultiply( inputMultiplier, inputMultiplier, objectCount, invObjectSize->GetData() );
		// The value of inputMultiplier will be used later.
		// Now it's allowed to overwrite values in inputDiff.
	}

	MathEngine().MultiplyMatrixByDiagMatrix( outputDiff, objectCount, objectSize, scale, inputDiff, inputDiffBlobs[0]->GetDataSize() );

	{
		CFloatHandle avgOfScaledOutDiff = buff.GetHandle();
		MathEngine().SumMatrixColumns( avgOfScaledOutDiff, inputDiff, objectCount, objectSize );
		MathEngine().VectorNegMultiply( avgOfScaledOutDiff, avgOfScaledOutDiff, objectCount, invObjectSize->GetData() );
		MathEngine().AddVectorToMatrixColumns( inputDiff, inputDiff, objectCount, objectSize, avgOfScaledOutDiff );
	}

	// Now reuse inputMultiplier.
	MathEngine().MultiplyDiagMatrixByMatrixAndAdd( 1, inputMultiplier, objectCount, input, objectSize, inputDiff );
	MathEngine().MultiplyDiagMatrixByMatrix( invSqrtVar, objectCount, inputDiff, objectSize, inputDiff, inputDiffBlobs[0]->GetDataSize() );
}

void CObjectNormalizationLayer::LearnOnce()
{
	const int objectCount = inputBlobs[0]->GetObjectCount();
	const int objectSize = inputBlobs[0]->GetObjectSize();
	const int dataSize = objectCount * objectSize;

	CFloatHandle outDiff = outputDiffBackup == nullptr ? outputDiffBlobs[0]->GetData() : outputDiffBackup->GetData();

	MathEngine().SumMatrixRowsAdd( 1, BiasDiff()->GetData(), outDiff, objectCount, objectSize );

	// If layer didn't call backward then we can rewrite outputDiffBlobs[0] because it isn't used elsewhere.
	// If layer called backward then we can rewrite outputDiffBackup because this call is the only place where it's used.
	// As a result, outDiff memory can be overwritten in any case.
	MathEngine().VectorEltwiseMultiply( normalizedInput->GetData(), outDiff, outDiff, dataSize );
	MathEngine().SumMatrixRowsAdd( 1, ScaleDiff()->GetData(), outDiff, objectCount, objectSize );
}

} // namespace NeoML
