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

import UIKit

/// Selected image detail controller
class PreviewViewController: UIViewController, UIScrollViewDelegate {
    
    @IBOutlet private weak var runButton: UIButton!
    @IBOutlet private weak var closeButton: UIButton!
    @IBOutlet private weak var imageView: UIImageView!
    @IBOutlet private weak var statusView: ClassificationStatusView!
    @IBOutlet private weak var scrollView: UIScrollView!
    var item: DocumentItem!
    var engine: ClassificationEngine!
    
    /// MARK: - LifeCycle
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
    }
    
    //MARK: - UI
    private func setupUI() {
        scrollView.minimumZoomScale = 1.0
        scrollView.maximumZoomScale = 5.0
        updateUI()
    }
    
    func updateUI() {
        statusView.showStatus(status: item.documentType)
        imageView.image = item.image
    }
    
    //MARK: - Actions
    @IBAction private func closeButtonTapped(_ sender: UIButton) {
        dismiss(animated: true, completion: nil)
    }
    
    @IBAction private func runButtonTapped(_ sender: UIButton) {
        engine.classify(item) {[weak self] in
            DispatchQueue.main.async {
                self?.updateUI()
            }
        }
    }
    
    //MARK: - UIScrollViewDelegate
    func viewForZooming(in scrollView: UIScrollView) -> UIView? {
        return imageView
    }
    
}
