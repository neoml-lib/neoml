# Функции активации

<!-- TOC -->

- [Функции активации](#функции-активации)
    - [Создание слоя](#создание-слоя)
    - [Виды слоёв](#виды-слоёв)

<!-- /TOC -->

Каждый из этих слоёв вычисляет определенную математическую функцию над своими входами.

## Создание слоя

```c++
CPtr<CBaseLayer> NEOML_API CreateActivationLayer( TActivationFunction type );
```

Создаёт слой, вычисляющий функцию активации `type`.

## Виды слоёв

```c++
enum TActivationFunction {
	AF_Linear = 0,
	AF_ELU,
	AF_ReLU,
	AF_LeakyReLU,
	AF_Abs,
	AF_Sigmoid,
	AF_Tanh,
	AF_HardTanh,
	AF_HardSigmoid,
	AF_Power,
	AF_HSwish,
	AF_GELU,

	AF_Count
};
```

Константа TActivationFunction | Имя класса | Функция активации
----------|-----------|--------------------
`AF_Linear` | [CLinearLayer](LinearLayer.md) | линейная функция активации: `ax + b`
`AF_ELU` | [CELULayer](ELULayer.md) | функция активации `ELU`
`AF_ReLU` | [CReLULayer](ReLULayer.md) | функция активации `ReLU`
`AF_LeakyReLU` | [CLeakyReLULayer](LeakyReLULayer.md) | функция активации `LeakyReLU`
`AF_Abs` | [CAbsLayer](AbsLayer.md) | функция активации `abs(x)`
`AF_Sigmoid` | [CSigmoidLayer](SigmoidLayer.md) | функция активации `sigmoid`
`AF_Tanh` | [CTanhLayer](TanhLayer.md) | функция активации `tanh`
`AF_HardTanh` | [CHardTanhLayer](HardTanhLayer.md) | функция активации `HardTanh`
`AF_HardSigmoid` | [CHardSigmoidLayer](HardSigmoidLayer.md) | функция активации `HardSigmoid`
`AF_Power` | [CPowerLayer](PowerLayer.md) | функция активации `pow(x, exp)`
`AF_HSwish` | [CHSwishLayer](HSwishLayer.md) | функция активации `h-swish`
`AF_GELU` | [CGELULayer](GELULayer.md) | функция активации `x * sigmoid(1.702 * x)`
`AF_Count` | | Вспомогательная константа, равная числу поддержанных функций активации.
