package com.neoml.test;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import java.io.InputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.lang.String;
import java.lang.Math;

import com.neoml.inference.NeoBlob;
import com.neoml.inference.NeoMathEngine;
import com.neoml.inference.NeoDnn;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ByteBuffer directBuffer;
        ByteBuffer inputDirectBuffer;
        ByteBuffer outputDirectBuffer;
        try {
            InputStream dnnStream = getAssets().open("testData/MobileNetV2Cifar10.cnnarch");

            int dnnSize = dnnStream.available();
            byte[] buffer = new byte[dnnSize];
            dnnStream.read(buffer);
            dnnStream.close();

            directBuffer = ByteBuffer.allocateDirect(dnnSize);
            directBuffer.order(ByteOrder.nativeOrder());
            directBuffer.put( buffer );
            directBuffer.rewind();

            InputStream inputStream = getAssets().open("testData/MobileNetV2Cifar10.input");
            int inputSize = inputStream.available();
            byte[] inputBuffer = new byte[inputSize];
            inputStream.read(inputBuffer);
            inputStream.close();

            inputDirectBuffer = ByteBuffer.allocateDirect(inputSize);
            inputDirectBuffer.order(ByteOrder.nativeOrder());
            inputDirectBuffer.put( inputBuffer );
            inputDirectBuffer.rewind();

            InputStream outputStream = getAssets().open("testData/MobileNetV2Cifar10.output");
            int outputSize = outputStream.available();
            byte[] outputBuffer = new byte[outputSize];
            outputStream.read(outputBuffer);
            outputStream.close();

            outputDirectBuffer = ByteBuffer.allocateDirect(outputSize);
            outputDirectBuffer.order(ByteOrder.nativeOrder());
            outputDirectBuffer.put( outputBuffer );
            outputDirectBuffer.rewind();

        } catch( IOException e ) {
            return;
        }

        try {
            NeoMathEngine mathEngine = NeoMathEngine.CreateCpuMathEngine(1);
            NeoBlob inputBlob = NeoBlob.CreateDnnBlob(mathEngine, NeoBlob.Type.FLOAT32, 1, 1, 32, 32, 1, 3);
            inputBlob.SetData(inputDirectBuffer);
            NeoDnn dnn = NeoDnn.CreateDnn(mathEngine, directBuffer);
            dnn.SetInputBlob( 0, inputBlob );

            dnn.Run();

            NeoBlob result = dnn.GetOutputBlob(0);

            ByteBuffer resultBuffer = result.GetData();

            float[] actual = new float[resultBuffer.capacity() / 4];
            resultBuffer.asFloatBuffer().get( actual );
            float[] expected = new float[outputDirectBuffer.capacity() / 4];
            outputDirectBuffer.asFloatBuffer().get( expected );

            for( int i = 0; i < expected.length; i++ ) {
                if( actual[i] - expected[i] < -1e-3 || actual[i] - expected[i] > 1e-3 ) {
                    Log.d( "ASSERT", "wrong answer!" );
                }
            }

        } catch( Exception e ) {
            Log.d( "EXCEPTION", e.toString() );
        }
    }
}
