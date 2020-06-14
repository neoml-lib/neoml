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

/// ListDataManager delagate
protocol ListDataManagerDelegate: AnyObject {
    func listDataManager(_ manager:ListDataManager, didUpdateItem item: DocumentItem)
}

//MARK: - ListDataManager
/// Handle data for ListViewController
class ListDataManager {
    
    let engine: ClassificationEngine!
    weak var delegate: ListDataManagerDelegate?
    private let fileManager = FileManager.default
    
    private lazy var items: [Int: DocumentItem] = [:]
    
    init(delegate: ListDataManagerDelegate, engine: ClassificationEngine) {
        self.delegate = delegate
        self.engine = engine
        self.engine.delegate = self
    }
            
    var itemsCount: Int {
        return items.count
    }
    
    func item(at index: Int) -> DocumentItem {
        return items[index]!
    }
    
    func index(for item: DocumentItem) -> Int? {
        return item.key
    }
    
    func update(item: DocumentItem, at index: Int) {
        items[index] = item
    }
    
    func loadItems() {
        loadSampleImageItems()
    }
    
    func updateItems() {
        engine.classify(Array(items.values))
    }

    /// Load all images from app data bundle and craete items
    private func loadSampleImageItems() {
        let bundleURL = Bundle.main.bundleURL
        let assetURL = bundleURL.appendingPathComponent("data")
        let imageURLs = imageUrls(for: assetURL).shuffled()
        
        var allItems: [Int: DocumentItem] = [:]
        for (index, url) in imageURLs.enumerated() {
            let image = UIImage(contentsOfFile: url.path)!
            allItems[index] = DocumentItem(image: image, key: index)
        }
        self.items = allItems
    }
    
    /// Search all image urls from data bundle
    /// - Parameter dirPath: ImagesBundle path
    private func imageUrls(for dirPath:URL) -> [URL] {
        var urls = [URL]()
        let contents = try! fileManager.contentsOfDirectory(at: dirPath,
                                                            includingPropertiesForKeys: [URLResourceKey.nameKey, URLResourceKey.isDirectoryKey],
                                                            options: .skipsHiddenFiles)
        let filtered = contents.filter{$0.hasDirectoryPath || $0.path.contains(".jpg")}
        filtered.forEach { item in
            !item.hasDirectoryPath ? urls.append(item) : urls.append(contentsOf: imageUrls(for: item))
        }
        return urls
    }

}

//MARK: - ClassificationEngineDelegate
extension ListDataManager: ClassificationEngineDelegate {
    func classificationEngine(_ engine: ClassificationEngine, didUpdate item: DocumentItem) {
        delegate?.listDataManager(self, didUpdateItem: item)
    }
}
