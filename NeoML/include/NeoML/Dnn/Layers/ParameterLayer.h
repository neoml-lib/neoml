/* Copyright © 2024 ABBYY Production LLC

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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// Layer with blob of trainable parameters ( nn.Parameter analogous )
class NEOML_API CParameterLayer : public CBaseLayer {
    NEOML_DNN_LAYER(CParameterLayer)
public:
    CParameterLayer( IMathEngine& mathEngine ) :
        CBaseLayer(mathEngine, "ParameterLayer", true) { paramBlobs.SetSize(1); };

    void Serialize(CArchive& archive) override;
    void SetBlob(CDnnBlob* _blob);
    void SetBlobDesc(const CBlobDesc& _desc);
    const CPtr<CDnnBlob>& GetBlob() const { return paramBlobs[0]; }
protected:
    void AllocateOutputBlobs() override;
    void Reshape() override;
    void RunOnce() override;
    void BackwardOnce() override;
    void LearnOnce() override;
private:
    CBlobDesc desc;
};

// To make it more convenient to create a class object
NEOML_API CParameterLayer* Parameter(CDnn& dnn, const char* name, CDnnBlob* blob);

} // namespace NeoML
