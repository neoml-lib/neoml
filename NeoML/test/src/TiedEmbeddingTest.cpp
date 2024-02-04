/* Copyright © 2024 ABBYY

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
#include <NeoML/NeoML.h>

using namespace NeoML;
using namespace NeoMLTest;

//----------------------------------------------------------------------------------------------------------------------

namespace NeoMLTest {

TEST(TiedEmbeddingTest, TiedEmbeddingTest)
{
	CRandom random(0x6543);
	CDnn net(random, MathEngine());

	const int seqLen = 100;
	const int batchSize = 1;
	const int vectorCount = 200;
	CPtr<CSourceLayer> data = Source(net, "data");
	CPtr<CDnnBlob> dataBlob = CDnnBlob::CreateDataBlob(MathEngine(), CT_Float, 1, seqLen, 8);
	for (int i = 0; i < dataBlob->GetDataSize(); ++i) {
		dataBlob->GetData().SetValueAt(i, random.UniformInt(0, vectorCount - 1));
	}
	data->SetBlob(dataBlob);

	CPtr<CCompositeLayer> compositeInner = new CCompositeLayer(net.GetMathEngine());
	compositeInner->SetName("compositeInner");

	CPtr<CMultichannelLookupLayer> lookup = new CMultichannelLookupLayer(MathEngine());
	CArray<CLookupDimension> lookupSize;
	lookupSize.SetSize(1);
	lookupSize[0].VectorSize = 8;
	lookupSize[0].VectorCount = vectorCount;
	lookup->SetDimensions(lookupSize);
	lookup->SetName({ "lookup" });
	CPtr<CDnnInitializer> embeddingInitializer = new CDnnUniformInitializer(random);
	lookup->Initialize(embeddingInitializer);
	compositeInner->SetInputMapping(*lookup);
	compositeInner->SetOutputMapping(*lookup);
	compositeInner->AddLayer(*lookup);

	CPtr<CCompositeLayer> composite = new CCompositeLayer(net.GetMathEngine());
	composite->SetName("composite");
	composite->Connect(*data);
	composite->AddLayer(*compositeInner);
	composite->SetInputMapping(*compositeInner);
	composite->SetOutputMapping(*compositeInner);
	net.AddLayer(*composite);


	CPtr<CTiedEmbeddingsLayer> tiedEmb = new CTiedEmbeddingsLayer(MathEngine());
	tiedEmb->SetEmbeddingsLayerName({ "composite", "compositeInner", "lookup" });
	net.AddLayer(*tiedEmb);
	tiedEmb->Connect(*data);

	CPtr<CSinkLayer> output1 = new CSinkLayer(MathEngine());
	output1->SetName("output1");
	net.AddLayer(*output1);
	output1->Connect(*tiedEmb);

	CPtr<CSinkLayer> output2 = new CSinkLayer(MathEngine());
	output2->SetName("output2");
	net.AddLayer(*output2);
	output2->Connect(*composite);
	net.RunOnce();

	ASSERT_EQ(dynamic_cast<const CCompositeLayer*>(const_cast<const CDnn&>(net).GetLayer({ "composite", "compositeInner" }).Ptr())->GetLayer("lookup"), lookup);
	ASSERT_EQ(dynamic_cast<CCompositeLayer*>(const_cast<CDnn&>(net).GetLayer({ "composite", "compositeInner" }).Ptr())->GetLayer("lookup"), lookup);
	ASSERT_EQ(dynamic_cast<CCompositeLayer*>(net.GetLayer("composite").Ptr())->GetLayer({ "compositeInner", "lookup" }), lookup);
	ASSERT_EQ(net.GetLayer({ "composite", "compositeInner", "lookup" }), lookup);
}

} // namespace NeoMLTest
