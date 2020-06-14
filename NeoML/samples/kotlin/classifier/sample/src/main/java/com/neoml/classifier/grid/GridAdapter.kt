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

package com.neoml.classifier.grid

import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.widget.AppCompatImageView
import androidx.core.view.ViewCompat
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide
import com.neoml.classifier.R
import com.neoml.classifier.model.ImageData
import com.neoml.classifier.model.Type


class GridAdapter(private val clickAction: (view: View, imageData: ImageData) -> Unit) :
    ListAdapter<ImageData, GridAdapter.ImageHolder>(ImagesDiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ImageHolder {
        val inflater = LayoutInflater.from(parent.context)
        return ImageHolder(clickAction, inflater.inflate(R.layout.grid_item, parent, false))
    }

    override fun onBindViewHolder(holder: ImageHolder, position: Int) {
        holder.bind(getItem(position))
    }

    class ImageHolder(val clickAction: (view: View, imageData: ImageData) -> Unit, private val v: View) : RecyclerView.ViewHolder(v),
        View.OnClickListener {
        init {
            itemView.setOnClickListener(this)
        }

        private lateinit var data: ImageData

        private val imageView = v.findViewById<AppCompatImageView>(R.id.photoImageView)

        override fun onClick(v: View) {
            ViewCompat.setTransitionName(imageView, data?.imagePath)
            clickAction(imageView, data)
        }

        fun bind(imageData: ImageData) {
            data = imageData
            Glide
                .with(itemView)
                .load(Uri.parse(imageData.imagePath))
                .dontAnimate()
                .centerCrop()
                .into(imageView)
            changeType(imageData.type)
        }

        private fun changeType(type: Type) {
            v.findViewById<ImageView>(R.id.bubbleIcon).setImageResource(type.drawableId)
            v.findViewById<TextView>(R.id.textClassificationTextView).setText(type.textId)
        }
    }
}