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

/// ListViewDataSourceProviderProtocol delegate
protocol ListViewDataSourceProviderDelegate: AnyObject {
    func dataProvider(_ provider: ListViewDataSourceProviderProtocol, didSelectItem item: DocumentItem)
}

protocol ListViewDataSourceProviderProtocol: UICollectionViewDataSource, UICollectionViewDelegate {}

//MARK: - ListViewDataSourceProvider
/// UICollectionView dataSource/delegate class
class ListViewDataSourceProvider: NSObject, ListViewDataSourceProviderProtocol {
    
    private let dataManager: ListDataManager
    weak var delegate: ListViewDataSourceProviderDelegate?
    private(set) var uiDelegate: ListCollectionViewUIDelegate!
    
    init(dataManager: ListDataManager, delegate: ListViewDataSourceProviderDelegate, uiDelegate: ListCollectionViewUIDelegate) {
        self.dataManager = dataManager
        self.delegate = delegate
        self.uiDelegate = uiDelegate
        super.init()
    }
}

//MARK: - UICollectionViewDataSource
extension ListViewDataSourceProvider: UICollectionViewDataSource {
    
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return dataManager.itemsCount
    }
    
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        if let cell = collectionView.dequeueReusableCell(withReuseIdentifier: String(describing: ListCollectionViewCell.self), for: indexPath) as? ListCollectionViewCell {
            let item = dataManager.item(at: indexPath.item)
            cell.configure(for: item)
            cell.updateUI()
            return cell
        }
        return UICollectionViewCell()
    }
}

//MARK: - UICollectionViewDelegate
extension ListViewDataSourceProvider: UICollectionViewDelegate {
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        let item = dataManager.item(at: indexPath.item)
        delegate?.dataProvider(self, didSelectItem: item)
    }
}
