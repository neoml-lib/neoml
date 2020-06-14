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

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.GridLayoutManager
import com.neoml.classifier.R
import com.neoml.classifier.detail.DetailFragment
import com.neoml.classifier.main.ActivityListener
import com.neoml.classifier.main.MainActivity
import com.neoml.classifier.model.ImageData
import com.neoml.classifier.model.MainModel
import kotlinx.android.synthetic.main.fragment_grid.*
import java.util.*


class GridFragment : Fragment(), Observer {

    companion object {
        fun createInstance() = GridFragment()
    }

    private lateinit var adapter: GridAdapter

    private val clickItemAction: (view: View, data: ImageData)-> Unit = { view, data ->
        (requireActivity() as MainActivity).supportFragmentManager
            .beginTransaction()
            .addSharedElement(view, data.imagePath)
            .addToBackStack(null)
            .replace(R.id.fragment_container, DetailFragment.createInstance(data))
            .commit()
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View = inflater.inflate(R.layout.fragment_grid, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        initRecyclerView()
        with((requireActivity() as? ActivityListener) ?: throw IllegalStateException()) {
            changeBottomButtons(isDetail = false)
            addObserver(this@GridFragment)
            bindRunAction()
            requestData()
        }

    }

    override fun update(o: Observable, arg: Any) {
        if (o is MainModel && arg is List<*>) {
            adapter.submitList(arg as List<ImageData>)
        }
    }

    private fun initRecyclerView() {
        adapter = GridAdapter(clickItemAction)
        recyclerView.layoutManager = GridLayoutManager(requireContext(), 3)
        recyclerView.adapter = adapter
        recyclerView.addItemDecoration(GridItemDecoration())
    }

    override fun onDestroyView() {
        super.onDestroyView()
        (requireActivity() as ActivityListener).deleteObserver(this)
    }

}
