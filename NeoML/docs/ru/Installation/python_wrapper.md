# Build the Python Wrapper

<!-- TOC -->

- [Перед сборкой](#перед-сборкой)
- [Windows](#windows)

<!-- /TOC -->

## Перед сборкой

Скачайте исходный код библиотеки из репозитория:

``` sh
git clone https://github.com/neoml-lib/neoml <path_to_src>
```

Создайте рабочую директорию рядом с исходным кодом:

``` sh
mkdir PythonBuild
cd PythonBuild
```

В приведённых примерах мы используем [стандартные команды и опции CMake](https://cmake.org/cmake/help/latest/index.html), с помощью которых задаётся архитектура, конфигурация сборки, компилятор и т.п. Поменяйте настройки под себя, если нужно.

## Windows
Процесс установки аналогичен инструкции по сборке С++ версии, но  нужно указать другой путь к Python-обертке, поскольку у нее свой CMakeLists файл. Он находится в директории **NeoML/Python**.
Чтобы создать проект, потребуется Microsoft Visual Studio версии 2015 или выше.

Пример команды для генерации проекта:

``` console 
cmake -G "Visual Studio 14 2015" -A <arch> <path_to_src>/NeoML/Python -DCMAKE_INSTALL_PREFIX=<install_path>
```

* \<arch> может принимать значение win32 или x64.
* \<install_path> путь к **PythonBuild** директории, созданной на предыдущем шаге.


Теперь можно открыть проект и собрать его с помощью Visual Studio, либо воспользоваться командой:

``` console
cmake --build . --target install --config <cfg>
```

* \<cfg> может принимать одно из следующих значений: Debug, Release, RelWithDebInfo, MinSizeRel.
