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

/// CollectionView with recognition images
class ListViewController: UIViewController, ListViewDataSourceProviderDelegate {

    @IBOutlet private weak var collectionView: UICollectionView!
    @IBOutlet private weak var cameraButton: UIButton!
    @IBOutlet private weak var photosButton: UIButton!
    @IBOutlet private weak var runButton: UIButton!
    
    private var dataSourceProvider: ListViewDataSourceProviderProtocol!
    private var dataManager: ListDataManager!
    private var selectedItem: DocumentItem?
    private let showDetailsSegueID = "showDetails"

    //MARK: - ViewController LifeCycle
    override func viewDidLoad() {
        super.viewDidLoad()
        collectionView.dataSource = dataSourceProvider
        collectionView.delegate = dataSourceProvider
        dataManager.loadItems()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        collectionView.reloadData()
    }
    
    // MARK: - Navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == showDetailsSegueID,
            let detailsPreviewController = segue.destination as? PreviewViewController,
            let item = selectedItem {
            detailsPreviewController.item = item
            detailsPreviewController.engine = dataManager.engine
        }
    }
    
    //MARK: - Managers/Delegate setup
    /// CollectionView dataSourece/Delegate setup
    /// - Parameter builder: creates classifaction engine
    func configure(with engine: ClassificationEngine) {
        dataManager = ListDataManager(delegate: self, engine: engine)
        dataSourceProvider = ListViewDataSourceProvider(dataManager: dataManager,
                                                               delegate: self,
                                                               uiDelegate: ListCollectionViewUIDelegate())
    }
        
    //MARK: - Actions
    @IBAction private func runButtonTapped(_ sender: UIButton) {
        dataManager.updateItems()
        collectionView.reloadData()
    }
    
    @IBAction private func cameraButtonTapped(_ sender: UIButton) {
        openMedia(type: .camera)
    }
    
    @IBAction private func photosButtonTapped(_ sender: UIButton) {
        openMedia(type: .photoLibrary)
    }
}

//MARK: - ImagePicker
extension ListViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        
    private func openMedia(type:UIImagePickerController.SourceType) {
        if UIImagePickerController.isSourceTypeAvailable(type){
            let picker = UIImagePickerController()
            picker.delegate = self;
            picker.sourceType = type
            present(picker, animated: true, completion: nil)
        }
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        let image: UIImage!
        if let editedImage = info[UIImagePickerController.InfoKey.editedImage] as? UIImage  {
            image = editedImage
        } else if let originalImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
            image = originalImage
        } else {
            dismiss(animated: true, completion: nil)
            return
        }
        let item = DocumentItem(image: image, key: 0)
        selectedItem = item
        dismiss(animated: true) { [unowned self] in
            self.performSegue(withIdentifier: self.showDetailsSegueID, sender: nil)
        }
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
}

//MARK: - Delagtes
extension ListViewController: ListDataManagerDelegate {
    
    //MARK: ListDataManagerDelegate
    func listDataManager(_ manager: ListDataManager, didUpdateItem item: DocumentItem) {
        if let index = manager.index(for: item) {
            DispatchQueue.main.async {
                let indexPathToUpdate = IndexPath(item: index, section: 0)
                guard self.collectionView.indexPathsForVisibleItems.contains(indexPathToUpdate) else { return }
                self.collectionView.reloadItems(at: [indexPathToUpdate])
            }
        }
    }

    // MARK: ListViewDataSourceProviderDelegate
    func dataProvider(_ provider: ListViewDataSourceProviderProtocol, didSelectItem item: DocumentItem) {
        selectedItem = item
        performSegue(withIdentifier: showDetailsSegueID, sender: nil)
    }
}

