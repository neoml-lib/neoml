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

#include <jni.h>
#include <stdio.h>
#include <random>
#include <NeoProxy/NeoProxy.h>

static_assert(sizeof(int) == sizeof(jint), "Incompatible INT size");
static_assert(sizeof(float) == sizeof(jfloat), "Incompatible FLOAT size");

static_assert(TDnnBlobType::DBT_Float == 1, "Incorrect enum value");
static_assert(TDnnBlobType::DBT_Int == 2, "Incorrect enum value");

static TDnnBlobType convertIntToDnnBlobType(jint t)
{
    return static_cast<TDnnBlobType>(t + 1);
}

static jint convertDnnBlobTypeToInt(TDnnBlobType t)
{
    return static_cast<jint>(t) - 1;
}

void throwJavaException(JNIEnv* env, const char* message)
{
    jclass jcls = env->FindClass("java/lang/RuntimeException");
    if( !env->ExceptionCheck() ) {
        env->ThrowNew(jcls, message);
    }
}

void throwJavaException(JNIEnv* env, const CDnnErrorInfo& error)
{
    throwJavaException(env, error.Description);
}

//------------------------------------------------------------------------------------------------------------
// NeoMathEngine

extern "C" JNIEXPORT jlong JNICALL
Java_com_neoml_inference_NeoMathEngine_createCpuMathEngine(JNIEnv* env, jobject, jint threads)
{
    CDnnErrorInfo error;
    const CDnnMathEngineDesc* mathEngineDesc = CreateCPUMathEngine(static_cast<int>(threads), &error);
    if( mathEngineDesc == 0 ) {
        throwJavaException(env, error);
        return 0;
    }
    return reinterpret_cast<long>(mathEngineDesc);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_neoml_inference_NeoMathEngine_createGpuMathEngine(JNIEnv* env, jobject)
{
    CDnnErrorInfo error;
    const CDnnMathEngineDesc* mathEngineDesc = CreateGPUMathEngine( &error );
    if( mathEngineDesc == 0 ) {
        throwJavaException(env, error);
        return 0;
    }
    return reinterpret_cast<long>(mathEngineDesc);
}

extern "C" JNIEXPORT void JNICALL
Java_com_neoml_inference_NeoMathEngine_destroyMathEngine(JNIEnv*, jobject, jlong nativeHandle)
{
    CDnnMathEngineDesc* mathEngineDesc = reinterpret_cast<CDnnMathEngineDesc*>( nativeHandle );
    if( mathEngineDesc != 0 ) {
        DestroyMathEngine( mathEngineDesc );
    }
}

//------------------------------------------------------------------------------------------------------------
// NeoBlob

extern "C" JNIEXPORT long JNICALL
Java_com_neoml_inference_NeoBlob_createDnnBlob(JNIEnv* env, jobject, jlong mathEngineNativeHandle, jint type,
    jint batchLength, jint batchWidth, jint height, jint width, jint depth, jint channelCount)
{
    CDnnErrorInfo error;
    CDnnMathEngineDesc* mathEngineDesc = reinterpret_cast<CDnnMathEngineDesc*>( mathEngineNativeHandle );
    const CDnnBlobDesc* blobDesc = CreateDnnBlob( mathEngineDesc, convertIntToDnnBlobType(type), batchLength,
        batchWidth, height, width, depth, channelCount, &error );
    if( blobDesc == 0 ) {
        throwJavaException(env, error);
        return 0;
    }
    return reinterpret_cast<long>(blobDesc);
}

extern "C" JNIEXPORT void JNICALL
Java_com_neoml_inference_NeoBlob_destroyDnnBlob(JNIEnv*, jobject, jlong nativeHandle)
{
    CDnnBlobDesc* blobDesc = reinterpret_cast<CDnnBlobDesc*>( nativeHandle );
    if( blobDesc != 0 ) {
        DestroyDnnBlob( blobDesc );
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_neoml_inference_NeoBlob_getType(JNIEnv*, jobject, jlong nativeHandle)
{
    CDnnBlobDesc* blobDesc = reinterpret_cast<CDnnBlobDesc*>( nativeHandle );
    return convertDnnBlobTypeToInt(blobDesc->Type);
}

extern "C" JNIEXPORT int JNICALL
Java_com_neoml_inference_NeoBlob_getBatchLength(JNIEnv*, jobject, jlong nativeHandle)
{
    CDnnBlobDesc* blobDesc = reinterpret_cast<CDnnBlobDesc*>( nativeHandle );
    return blobDesc->BatchLength;
}

extern "C" JNIEXPORT int JNICALL
Java_com_neoml_inference_NeoBlob_getBatchWidth(JNIEnv*, jobject, jlong nativeHandle)
{
    CDnnBlobDesc* blobDesc = reinterpret_cast<CDnnBlobDesc*>( nativeHandle );
    return blobDesc->BatchWidth;
}

extern "C" JNIEXPORT int JNICALL
Java_com_neoml_inference_NeoBlob_getHeight(JNIEnv*, jobject, jlong nativeHandle)
{
    CDnnBlobDesc* blobDesc = reinterpret_cast<CDnnBlobDesc*>( nativeHandle );
    return blobDesc->Height;
}

extern "C" JNIEXPORT int JNICALL
Java_com_neoml_inference_NeoBlob_getWidth(JNIEnv*, jobject, jlong nativeHandle)
{
    CDnnBlobDesc* blobDesc = reinterpret_cast<CDnnBlobDesc*>( nativeHandle );
    return blobDesc->Width;
}

extern "C" JNIEXPORT int JNICALL
Java_com_neoml_inference_NeoBlob_getDepth(JNIEnv*, jobject, jlong nativeHandle)
{
    CDnnBlobDesc* blobDesc = reinterpret_cast<CDnnBlobDesc*>( nativeHandle );
    return blobDesc->Depth;
}

extern "C" JNIEXPORT int JNICALL
Java_com_neoml_inference_NeoBlob_getChannelsCount(JNIEnv*, jobject, jlong nativeHandle)
{
    CDnnBlobDesc* blobDesc = reinterpret_cast<CDnnBlobDesc*>( nativeHandle );
    return blobDesc->ChannelCount;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_neoml_inference_NeoBlob_getDataSize(JNIEnv*, jobject, jlong nativeHandle)
{
    CDnnBlobDesc* blobDesc = reinterpret_cast<CDnnBlobDesc*>( nativeHandle );
    return blobDesc->DataSize;
}

extern "C" JNIEXPORT void JNICALL
Java_com_neoml_inference_NeoBlob_setData(JNIEnv* env, jobject, jlong nativeHandle, jobject buffer)
{
    CDnnBlobDesc* blobDesc = reinterpret_cast<CDnnBlobDesc*>(nativeHandle);
    CDnnErrorInfo error;
    if( !CopyToBlob(blobDesc, env->GetDirectBufferAddress(buffer), &error) ) {
        throwJavaException(env, error);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_neoml_inference_NeoBlob_getData(JNIEnv* env, jobject, jlong nativeHandle, jobject buffer)
{
    CDnnBlobDesc* blobDesc = reinterpret_cast<CDnnBlobDesc*>(nativeHandle);
    CDnnErrorInfo error;
    if( !CopyFromBlob(env->GetDirectBufferAddress(buffer), blobDesc, &error) ) {
        throwJavaException(env, error);
    }
}

//------------------------------------------------------------------------------------------------------------
// NeoDnn

extern "C" JNIEXPORT jlong JNICALL
Java_com_neoml_inference_NeoDnn_createDnn(JNIEnv* env, jobject, jlong mathEngineNativeHandle, jobject buffer)
{
    CDnnErrorInfo error;
    CDnnMathEngineDesc* mathEngineDesc = reinterpret_cast<CDnnMathEngineDesc*>( mathEngineNativeHandle );
    auto data = env->GetDirectBufferAddress(buffer);
    int size = static_cast<int>(env->GetDirectBufferCapacity(buffer));
    const CDnnDesc* dnnDesc = CreateDnnFromBuffer(mathEngineDesc, data, size, &error);
    if( dnnDesc == 0 ) {
        throwJavaException(env, error);
        return 0;
    }
    return reinterpret_cast<long>(dnnDesc);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_neoml_inference_NeoDnn_createDnnFromOnnx(JNIEnv* env, jobject, jlong mathEngineNativeHandle, jobject buffer)
{
    CDnnErrorInfo error;
    CDnnMathEngineDesc* mathEngineDesc = reinterpret_cast<CDnnMathEngineDesc*>( mathEngineNativeHandle );
    auto data = env->GetDirectBufferAddress(buffer);
    int size = static_cast<int>(env->GetDirectBufferCapacity(buffer));
    const CDnnDesc* dnnDesc = CreateDnnFromOnnxBuffer(mathEngineDesc, data, size, &error);
    if( dnnDesc == 0 ) {
        throwJavaException(env, error);
        return 0;
    }
    return reinterpret_cast<long>(dnnDesc);
}

extern "C" JNIEXPORT void JNICALL
Java_com_neoml_inference_NeoDnn_destroyDnn(JNIEnv*, jobject, jlong nativeHandle)
{
    CDnnDesc* dnnDesc = reinterpret_cast<CDnnDesc*>( nativeHandle );
    if( dnnDesc != 0 ) {
        DestroyDnn(dnnDesc);
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_neoml_inference_NeoDnn_getInputCount(JNIEnv*, jobject, jlong nativeHandle)
{
    CDnnDesc* dnnDesc = reinterpret_cast<CDnnDesc*>( nativeHandle );
    return dnnDesc->InputCount;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_neoml_inference_NeoDnn_getInputName(JNIEnv* env, jobject, jlong nativeHandle, jint index)
{
    CDnnDesc* dnnDesc = reinterpret_cast<CDnnDesc*>( nativeHandle );
    CDnnErrorInfo error;
    const char* name = GetInputName(dnnDesc, index, &error);
    if( name == 0 ) {
        throwJavaException(env, error);
        return nullptr;
    }
    return env->NewStringUTF(name);
}

extern "C" JNIEXPORT void JNICALL
Java_com_neoml_inference_NeoDnn_setInputBlob(JNIEnv* env, jobject, jlong nativeHandle, jint index, jlong blobNativeHandle)
{
    CDnnDesc* dnnDesc = reinterpret_cast<CDnnDesc*>( nativeHandle );
    CDnnBlobDesc* blobDesc = reinterpret_cast<CDnnBlobDesc*>( blobNativeHandle );
    CDnnErrorInfo error;
    if( !SetInputBlob(dnnDesc, index, blobDesc, &error) ) {
        throwJavaException(env, error);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_neoml_inference_NeoDnn_run(JNIEnv* env, jobject, jlong nativeHandle)
{
    CDnnDesc* dnnDesc = reinterpret_cast<CDnnDesc*>( nativeHandle );
    CDnnErrorInfo error;
    if( !DnnRunOnce(dnnDesc, &error) ) {
        throwJavaException(env, error);
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_neoml_inference_NeoDnn_getOutputCount(JNIEnv*, jobject, jlong nativeHandle)
{
    CDnnDesc* dnnDesc = reinterpret_cast<CDnnDesc*>( nativeHandle );
    return dnnDesc->OutputCount;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_neoml_inference_NeoDnn_getOutputName(JNIEnv* env, jobject, jlong nativeHandle, jint index)
{
    CDnnErrorInfo error;
    CDnnDesc* dnnDesc = reinterpret_cast<CDnnDesc*>( nativeHandle );
    const char* name = GetOutputName(dnnDesc, index, &error);
    if( name == 0 ) {
        throwJavaException(env, error);
        return nullptr;
    }
    return env->NewStringUTF(name);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_neoml_inference_NeoDnn_getOutputBlob(JNIEnv* env, jobject, jlong nativeHandle, jint index)
{
    CDnnErrorInfo error;
    CDnnDesc* dnnDesc = reinterpret_cast<CDnnDesc*>( nativeHandle );
    const CDnnBlobDesc* blobDesc = GetOutputBlob(dnnDesc, index, &error);
    if( blobDesc == 0 ) {
        throwJavaException(env, error);
        return 0;
    }
    return reinterpret_cast<jlong>( blobDesc );
}
