# Иерархическая кластеризация CHierarchicalClustering

<!-- TOC -->

- [Иерархическая кластеризация CHierarchicalClustering](#иерархическая-кластеризация-chierarchicalclustering)
	- [Параметры](#параметры)
	- [Пример](#пример)

<!-- /TOC -->

Иерархическая кластеризация в **NeoML** представляет собой наивную реализацию восходящей версии алгоритма. 

В исходном состоянии каждому элементу множества соответствует отдельный кластер; далее начинается процесс объединения кластеров. В процессе объединения сливаются два наименее удалённых друг от друга кластера. Когда достигнуто нужное количество кластеров или все кластеры достаточно удалены друг от друга, процесс останавливается.

В **NeoML** алгоритм реализован классом `CHierarchicalClustering`, который предоставляет интерфейс `IClustering`. Кластеризация производится с помощью его метода `Clusterize`.

## Параметры

Параметры кластеризации описываются структурой `CHierarchicalClustering::CParam`.

- *DistanceType* — используемая функция расстояния;
- *MaxClustersDistance* — максимальное допустимое расстояние для склеивания двух кластеров;
- *MinClustersCount* — минимальное количество кластеров в результате.

## Пример

В данном примере набор данных разбивается на два кластера:

```c++
void Clusterize( const IClusteringData& data, CClusteringResult& result )
{
	CHierarchicalClustering::CParam params;
	params.DistanceType = DF_Euclid;
	params.MinClustersCount = 2;
	params.MaxClustersDistance = 10.f;

	CHierarchicalClustering hierarchical( params );
	hierarchical.Clusterize( data, result );
}
```
