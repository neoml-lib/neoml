# Сборка полной C++ версии

<!-- TOC -->

- [Перед сборкой](#перед-сборкой)
- [Windows](#windows)
- [Linux/macOS](#linux/macos)
- [Android](#android)
- [iOS](#ios)
- [Troubleshooting](#troubleshooting)

<!-- /TOC -->

## Перед сборкой

Скачайте исходный код библиотеки из репозитория:

``` sh
git clone https://github.com/neoml-lib/neoml <path_to_src>
```

Создайте рабочую директорию рядом с исходным кодом:

``` sh
mkdir Build
cd Build
```

В приведённых примерах мы используем [стандартные команды и опции CMake](https://cmake.org/cmake/help/latest/index.html), с помощью которых задаётся архитектура, конфигурация сборки, компилятор и т.п. Поменяйте настройки под себя, если нужно.

## Windows

Чтобы создать проект, потребуется Microsoft Visual Studio версии 2015 или выше.

Пример команды для генерации проекта:

``` console 
cmake -G "Visual Studio 14 2015" -A <arch> <path_to_src>/NeoML -DCMAKE_INSTALL_PREFIX=<install_path>
```

* \<arch> может принимать значение win32 или x64.
* \<install_path> путь к **Build** директории, созданной на предыдущем шаге.


Теперь можно открыть проект и собрать его с помощью Visual Studio, либо воспользоваться командой:

``` console
cmake --build . --target install --config <cfg>
```

* \<cfg> может принимать одно из следующих значений: Debug, Release, RelWithDebInfo, MinSizeRel.

Результатами сборки будут три динамические библиотеки: **NeoML.dll**, **NeoMathEngine.dll** и **NeoOnnx.dll**.

## Linux/macOS

Сгенерируйте правила сборки для Ninja:

``` console
cmake -G Ninja <path_to_src>/NeoML -DCMAKE_BUILD_TYPE=<cfg> -DCMAKE_INSTALL_PREFIX=<install_path>
```

* \<cfg> может принимать одно из следующих значений: Debug, Release, RelWithDebInfo, MinSizeRel.

Теперь соберите проект и установите собранные библиотеки в место назначения командой:

``` console
cmake --build . --target install
```
Результатами сборки будут три динамические библиотеки: **libNeoML.so**, **libNeoMathEngine.so**, **libNeoOnnx.so** для Linux и **libNeoML.dylib**, **libNeoMathEngine.dylib**, **libNeoOnnx.dylib** для MacOS.

## Android

Собрать библиотеку для Android можно на хост-машине, где установлена ОС Windows, Linux или macOS. Сначала установите [Android NDK](https://developer.android.com/ndk/downloads). 

Пример команды для генерации правил сборки для Ninja в качестве бекэнда:

``` console
cmake -G Ninja <path_to_src>/NeoML -DCMAKE_TOOLCHAIN_FILE=<path_to_ndk>/build/cmake/android.toolchain.cmake -DANDROID_ABI=<abi> -DCMAKE_INSTALL_PREFIX=<install_path> -DCMAKE_BUILD_TYPE=<cfg>
```

* \<abi> может принимать одно из следующих значений: armeabi-v7a, arm64-v8a, x86, x86_64.
* \<cfg> может принимать одно из следующих значений: Debug, Release, RelWithDebInfo, MinSizeRel.

Теперь соберите проект и установите собранные библиотеки в место назначения командой:

``` console
cmake --build . --target install
```

Результатами сборки будут три динамические библиотеки **libNeoML.so**, **libNeoMathEngine.so** и **libNeoOnnx.so**.

## iOS

Чтобы создать проект, понадобится Apple Xсode. В директории cmake в составе библиотеки есть toolchain-файл, который содержит настройки сборки для архитектур arm64 и x86_64. Используйте его для генерации проекта Xcode:

``` console
cmake -G Xcode <path_to_src>/NeoML -DCMAKE_TOOLCHAIN_FILE=<path_to_src>/cmake/ios.toolchain.cmake -DIOS_ARCH=<arch> -DCMAKE_INSTALL_PREFIX=<install_path>
```

* \<arch> может принимать значения arm64 или x86_64.

Теперь можно открыть и собрать проект с помощью Xcode, либо выполнить команду:

``` console
cmake --build . --target install --config <cfg>
```

* \<cfg> может принимать одно из следующих значений: Debug, Release, RelWithDebInfo, MinSizeRel.

Результатами сборки будут три фреймворка: **NeoML.framework**, **NeoMathEngine.framework** и **NeoOnnx.framework**.

## Troubleshooting

**Protobuf**

Иногда на ОС Windows, CMake не может найти путь к библиотеке Protobuf во время генерации проекта. Чтобы это исправить, можно самому указать путь к корневой папке библиотеке с помощью добавления аргумента `-DCMAKE_PREFIX_PATH=<path_to_Protobuf>` к команде CMake, которая генерирует проект.

В этом случае, команда CMake выглядит следующим образом:
``` console 
cmake -G "Visual Studio 14 2015" -A <arch> <path_to_src>/NeoML -DCMAKE_INSTALL_PREFIX=<install_path> -DCMAKE_PREFIX_PATH=<path_to_Protobuf>
```