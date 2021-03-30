#include <jni.h>
#include <string>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <TestFixture.h>

extern "C" JNIEXPORT void
JNICALL Java_com_neomathengine_inferencetest_MainActivity_RunTests( JNIEnv *env, jobject, jobject _assetManager )
{
    int argc = 2;
    char* argv[2] = {"--MathEngine=cpu", "--gtest_filter=*Qrnn*"};
    NeoMLTest::RunTests( argc, argv );
}
