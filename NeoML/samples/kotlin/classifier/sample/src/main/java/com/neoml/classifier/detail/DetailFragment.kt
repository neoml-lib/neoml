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

package com.neoml.classifier.detail

import android.graphics.drawable.Drawable
import android.net.Uri
import android.os.Bundle
import android.transition.TransitionInflater
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.view.ViewCompat
import androidx.fragment.app.Fragment
import com.bumptech.glide.Glide
import com.bumptech.glide.load.DataSource
import com.bumptech.glide.load.DecodeFormat
import com.bumptech.glide.load.engine.GlideException
import com.bumptech.glide.request.RequestListener
import com.neoml.classifier.R
import com.neoml.classifier.main.ActivityListener
import com.neoml.classifier.model.ImageData
import com.neoml.classifier.model.MainModel
import kotlinx.android.synthetic.main.fragment_detail.*
import com.bumptech.glide.request.target.Target
import java.util.*


class DetailFragment : Fragment(), Observer {

    companion object {
        private const val DATA_KEY = "data_key"

        fun createInstance(imageData: ImageData?): DetailFragment {
            val fragment = DetailFragment()
            val bundle = Bundle()
            bundle.putParcelable(DATA_KEY, imageData)
            fragment.arguments = bundle
            return fragment
        }
    }

    private val data: ImageData? by lazy {
        this.requireArguments().getParcelable<ImageData>(DATA_KEY)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        sharedElementEnterTransition = TransitionInflater
            .from(requireContext())
            .inflateTransition(R.transition.enter_detail_transition)
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View = inflater.inflate(R.layout.fragment_detail, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        ViewCompat.setTransitionName(detailImageView, data?.imagePath)
        (requireActivity() as ActivityListener).bindRunAction(data)

        data?.let(::updateContentOnViews)
    }

    private fun updateContentOnViews(data: ImageData) {
        detailBubbleIcon.setImageResource(data.type.drawableId)
        detailTextClassificationTextView.setText(data.type.textId)

        Glide.with(this)
            .load(Uri.parse(data.imagePath))
            .format(DecodeFormat.PREFER_ARGB_8888)
            .dontTransform()
            .encodeQuality(100)
            .dontAnimate()
            .listener(object : RequestListener<Drawable> {
                override fun onLoadFailed(
                    error: GlideException?,
                    model: Any?,
                    target: Target<Drawable>?,
                    isFirstResource: Boolean
                ): Boolean {
                    startPostponedEnterTransition()
                    return false
                }

                override fun onResourceReady(
                    resource: Drawable?,
                    model: Any?,
                    target: Target<Drawable>?,
                    dataSource: DataSource?,
                    isFirstResource: Boolean
                ): Boolean {
                    startPostponedEnterTransition()
                    return false
                }
            })
            .into(detailImageView)
    }

    override fun update(observable: Observable, arg: Any) {
        if (observable is MainModel && arg is List<*>) {
            (arg as List<ImageData>)
                .filter { it.imagePath == data?.imagePath }
                .forEach { it.let(::updateContentOnViews) }
        }
    }

    override fun onDetach() {
        super.onDetach()
        (requireActivity() as ActivityListener).deleteObserver(this)
    }

    override fun onResume() {
        super.onResume()
        with((requireContext() as? ActivityListener) ?: throw IllegalStateException()) {
            changeBottomButtons(isDetail = true)
            addObserver(this@DetailFragment)
            bindRunAction(imageData = data)
        }
    }

}