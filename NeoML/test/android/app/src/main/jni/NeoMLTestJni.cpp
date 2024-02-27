#include <jni.h>
#include <TestFixture.h>

extern "C"
JNIEXPORT void JNICALL Java_com_neoml_test_MainActivity_RunTests( JNIEnv *, jobject, jobject assetManager )
{
    int argc = 2;
    char* argv[2] = {"--MathEngine=cpu", "--gtest_filter=*"};
    NeoMLTest::RunTests( argc, argv, assetManager );
}
