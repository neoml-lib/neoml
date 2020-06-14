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

package com.neoml.classifier.main

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat.requestPermissions
import androidx.core.app.ActivityCompat.shouldShowRequestPermissionRationale
import androidx.core.content.ContextCompat.checkSelfPermission
import androidx.core.content.FileProvider
import com.neoml.classifier.R
import com.neoml.classifier.grid.GridFragment
import com.neoml.classifier.model.ImageData
import com.neoml.classifier.model.MainModel
import com.neoml.classifier.provider.Provider
import kotlinx.android.synthetic.main.activity_main.*
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*


class MainActivity : AppCompatActivity(), ActivityListener {

    companion object {
        const val CAMERA_REQUEST_CODE = 1
        const val GALLERY_REQUEST_CODE = 2
        const val REQUEST_PERMISSIONS_CAMERA = 3
    }

    private val model: MainModel by lazy { MainModel() }

    private val cameraPermission = Manifest.permission.CAMERA

    private val takePictureFromGalleryAction: (view: View) -> Unit =
        {
            startActivityForResult(
                Intent(
                    Intent.ACTION_PICK,
                    MediaStore.Images.Media.INTERNAL_CONTENT_URI
                ), GALLERY_REQUEST_CODE
            )
        }

    private var pictureUriFromCamera: Uri? = null

    private val takePictureFromCameraAction: (view: View) -> Unit =
        {
            val checkSelfPermission = checkSelfPermission(this, cameraPermission) != PackageManager.PERMISSION_GRANTED
            if (checkSelfPermission) {
                if (shouldShowRequestPermissionRationale(this, cameraPermission)) {
                    takePhotoFromCamera()
                } else {
                    requestPermissions(this, arrayOf(cameraPermission), REQUEST_PERMISSIONS_CAMERA)
                }
            } else {
                takePhotoFromCamera()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        if (savedInstanceState == null) {
            openGridFragment()
            gallery_button.setOnClickListener(takePictureFromGalleryAction)
            camera_button.setOnClickListener(takePictureFromCameraAction)
        }
    }

    @Throws(IOException::class)
    private fun takePhotoFromCamera() {
        val timeStamp: String = SimpleDateFormat("yyMMdd_HHmmss").format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        val file = File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir)
        pictureUriFromCamera = FileProvider.getUriForFile(
            this@MainActivity,
            Provider.getAuthority(this@MainActivity),
            file
        )

        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, pictureUriFromCamera)
        takePictureIntent.resolveActivity(packageManager)?.also {
            startActivityForResult(takePictureIntent, CAMERA_REQUEST_CODE)
        }
    }

    private fun openGridFragment() {
        this.supportFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, GridFragment.createInstance(), null)
            .commit()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        when {
            resultCode == RESULT_OK && requestCode == GALLERY_REQUEST_CODE -> {
                model.addImageFromGallery(data?.data.toString())
                model.requestNonHandledData()
            }
            resultCode == RESULT_OK && requestCode == CAMERA_REQUEST_CODE -> {
                model.addImageFromGallery(pictureUriFromCamera.toString())
                model.requestNonHandledData()
            }
            else -> {
                super.onActivityResult(requestCode, resultCode, data)
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        when (requestCode) {
            REQUEST_PERMISSIONS_CAMERA -> {
                if ((grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                    takePhotoFromCamera()
                }
            }
            else -> {
                super.onRequestPermissionsResult(requestCode, permissions, grantResults)
            }
        }
    }

    override fun changeBottomButtons(isDetail: Boolean) {
        val visibility = if (isDetail) View.GONE else View.VISIBLE
        camera_button.visibility = visibility
        gallery_button.visibility = visibility
    }

    override fun bindRunAction(imageData: ImageData?) {
        run_button.setOnClickListener {
            if (imageData == null) {
                model.classifyAllImages()
            } else {
                model.classify(
                    imagePath = imageData.imagePath,
                    isSingleRequest = true
                )
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        model.destroy()
    }

    override fun requestData() {
        model.requestNonHandledData()
    }

    override fun addObserver(observer: Observer) {
        model.addObserver(observer)
    }

    override fun deleteObserver(observer: Observer) {
        model.deleteObserver(observer)
    }

}
