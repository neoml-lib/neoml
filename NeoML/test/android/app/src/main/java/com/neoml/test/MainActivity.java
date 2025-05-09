package com.neoml.test;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.widget.TextView;
import android.support.v7.app.AppCompatActivity;
import android.view.Menu;
import android.view.MenuItem;

import android.os.AsyncTask;
import android.os.ParcelFileDescriptor;

import android.system.Os;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class MainActivity extends AppCompatActivity {
    class LogTask extends AsyncTask<ParcelFileDescriptor, String, Void> {
        @Override
        protected Void doInBackground(ParcelFileDescriptor... fds) {
            try {
                ParcelFileDescriptor.AutoCloseInputStream stream =
                        new ParcelFileDescriptor.AutoCloseInputStream(fds[0]);
                BufferedReader buffReader = new BufferedReader(new InputStreamReader(stream));
                String line;
                while ((line = buffReader.readLine()) != null) {
                    android.util.Log.d(Tag, line);
                    publishProgress(line); // pass the data to onProgressUpdate
                }
            } catch (Exception exception) {
                android.util.Log.e(Tag, "Except in log thread", exception);
            }
            return null;
        }

        @Override
        protected void onProgressUpdate(String... log) {
            TextView tv = (TextView) findViewById(R.id.logView);
            for (String line: log) {
                tv.append(line);
                tv.append("\n");
            }
        }
    }
    public native void RunTests( Object assetManager );

    static AssetManager assetManager;
    static Context context;
    static String Tag = "NeoMLTest";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        assetManager = getAssets();
        context = getApplicationContext();

        setContentView(R.layout.activity_main);
        // Example of a call to a native method
        System.loadLibrary("NeoMLTestJni");

        try {
            ParcelFileDescriptor pipe[] = ParcelFileDescriptor.createPipe();
            Os.dup2(pipe[1].getFileDescriptor(), 1); // stdout
            Os.dup2(pipe[1].getFileDescriptor(), 2); // stderr
            pipe[1].close();

            new LogTask().execute(pipe[0]);

            new Thread(new Runnable() {
                public void run() {
                    RunTests(assetManager);
                }
            }).start();

        } catch (Exception exception) {
            android.util.Log.e(Tag, "Except", exception);
        }
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}
