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
#ifdef NEOML_USE_FINEOBJ
#include <FineObj.h>
#endif
#include <NeoOnnx/NeoOnnx.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Random.h>
#include <iostream>

using namespace NeoML;

int ConvertOnnx2NeoML( const char* inputOnnxFilename, const char* outputDnnArchiveFileName )
{
    IMathEngine& mathEng = GetDefaultCpuMathEngine();
    CRandom random( 0x123 );
    CDnn net( random, mathEng );
    try {
        NeoOnnx::CImportSettings importSettings;
        NeoOnnx::CImportedModelInfo modelInfo;
        NeoOnnx::LoadFromOnnx( inputOnnxFilename, importSettings, net, modelInfo );
        {
            CArchiveFile file( outputDnnArchiveFileName, CArchive::store );
            CArchive archive( &file, CArchive::SD_Storing );
            archive.Serialize( net );
        }
        return 0;
    } catch( std::exception& exc ) {
        std::cout << "Exception" << std::endl;
        std::cout << exc.what() << std::endl;
        return 1;
    }
}
static const char* Help = 
"Usage:\n$ Onnx2NeoML <path to some existing model.onnx> <path to new saved model.dnnarchive>\n"
"Use this tool to convert your ONNX model "
"into the archived NeoML::CDnn. To be converted your model "
"must use only supported NeoOnnx operators.";

#ifdef NEOML_USE_FINEOBJ
int FineMain( int argc, wchar_t* argv[] )
#else
int main( int argc, char* argv[] )
#endif
{
    if( argc == 1 ) {
        std::cout << Help << std::endl;
        return 0;
    };
    if( argc != 3 ) {
        std::cout << "Error, expected 2 args, got " << argc - 1 << std::endl;
        std::cout << Help << std::endl;
        return 1;
    }
#ifdef NEOML_USE_FINEOBJ
    CString inpathStr = CUnicodeString( argv[1] ).CreateString();
    CString outpathStr = CUnicodeString( argv[2] ).CreateString();
    const char* inpath = inpathStr.Ptr();
    const char* outpath = outpathStr.Ptr();
#else
    const char* inpath = argv[1];
    const char* outpath = argv[2];
#endif
    return ConvertOnnx2NeoML( inpath, outpath );
}