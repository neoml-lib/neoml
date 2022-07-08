# Библиотека NeoOnnx

<!-- TOC -->
- [Библиотека NeoOnnx](#библиотека-neoonnx)
    - [API](#api)
        - [Загрузка сети](#загрузка-сети)
    - [Сборка](#сборка)
    - [Реализация](#реализация)
    - [Поддержка мобильных платформ](#поддержка-мобильных-платформ)
<!-- /TOC -->

Библиотека **NeoOnnx** предоставляет возможность загружать нейронные сети, сериализованные в формате ONNX.

## API

### Загрузка сети

```c++
#include <NeoOnnx/NeoOnnx.h>

NEOONNX_API void LoadFromOnnx( const char* fileName, const CImportSettings& settings,
    NeoML::CDnn& dnn, CImportedModelInfo& info );
NEOONNX_API void LoadFromOnnx( const void* buffer, int bufferSize, const CImportSettings& settings,
    NeoML::CDnn& dnn, CImportedModelInfo& info );
```

Загружает сеть из файла или из буфера.

Для каждого входа сети в `dnn` будет создан `CSourceLayer` с таким же именем. Для каждого такого слоя будет выделен блоб размера, указанного в ONNX-модели. Также имена входов будут добавлены в массив `inputs`. Входы сети с инициализаторами будут проигнорированы.

Для каждого выхода сети в `dnn` будет создан `CSinkLayer` с таким же именем. Также имена выходов будут добавлены в массив `outputs`.

Информация о входах и выходах, а также `metadata_props` модели будет записана в `info`.

## Сборка

Библиотека собирается автоматически вместе с **NeoML**.

## Реализация

Используется 9 версия (opset version) протокола ONNX. Поддержаны основные операции сверточных сетей, LSTM, основные функции активации.

## Поддержка мобильных платформ

См. методы загрузки сетей в [Objective-C](../en/Wrappers/ObjectiveC.md) и [Java](../en/Wrappers/Java.md) интерфейсах.
