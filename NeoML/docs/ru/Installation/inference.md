# Сборка версии для прямого прохода нейронных сетей

<!-- TOC -->

- [Перед сборкой](#перед-сборкой)
- [Java](#java)
- [Objective-C](#objective-c)

<!-- /TOC -->

## Перед сборкой

Скачайте исходный код библиотеки из репозитория:

``` sh
git clone https://github.com/neoml-lib/neoml <path_to_src>
```

### Java

В каталоге <path_to_src>/NeoML/Java находится проект, который позволяет собрать Android-библиотеку с помощью Android Studio. Кроме того, это можно сделать с помощью gradle в консоли. Для этого перейдите в соответствующую директорию с проектом:

``` console
cd <path_to_src>/NeoML/Java
```
Не забудьте установить переменную среды JAVA_HOME для указания директории, в которой установлена Java, а также переменную ANDROID_HOME, в которой должен быть указан путь к Android SDK.

Если сборка осуществляется на Windows, то выполните команду:

``` console
gradlew.bat "inference:assembleRelease"
```

Для macOS/Linux выполните:

``` console
./gradlew "inference:assembleRelease"
```

В директории <path_to_src>/NeoML/Java/inference/build/outputs/aar появится файл **NeoInference.aar**.

### Objective-C

Для сборки Objective-C библиотеки используются [стандартные команды и опции CMake](https://cmake.org/cmake/help/latest/index.html).

Создайте рабочую директорию:

``` sh
mkdir Build
cd Build
```

Чтобы создать проект, понадобится Apple Xcode. В директории cmake в составе библиотеки есть toolchain-файл, который содержит настройки сборки для архитектур arm64 и x86_64. Используйте его для генерации проекта Xcode:

``` console
cmake -G Xcode <path_to_src>/NeoML/objc -DCMAKE_TOOLCHAIN_FILE=<path_to_src>/cmake/ios.toolchain.cmake -DIOS_ARCH=<arch> -DCMAKE_INSTALL_PREFIX=<install_path>
```

* \<arch> может принимать значения arm64 или x86_64.

Теперь можно открыть и собрать проект с помощью Xcode, либо выполнить команду:

``` console
cmake --build . --target install --config <cfg>
```

* \<cfg> может принимать одно из следующих значений: Debug, Release, RelWithDebInfo, MinSizeRel.

Результатом сборки будет фреймворк **NeoInference.framework**.
