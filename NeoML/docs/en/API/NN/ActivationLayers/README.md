# Activation functions

<!-- TOC -->

- [Activation functions](#activation-functions)
	- [Creating a layer](#creating-a-layer)
		- [Activation parameters holder](#activation-parameters-holder)
	- [Layer types](#layer-types)

<!-- /TOC -->

This section describes the layers that calculate the value of specified activation functions for each of their inputs.


## Creating a layer

```c++
CPtr<CBaseLayer> NEOML_API CreateActivationLayer( const CActivationDesc& activation );
```

Creates a layer that calculates the activation function specified by the `activation` parameter.

### Activation parameters holder
To create a layer, one must specify its type and, optionally, parameters. All configurable activation layers have a `CParam` structure with all possible settings.

```c++
class CActivationDesc {
public:
	// For non-parametrized activations or default parameters
	CActivationDesc( TActivationFunction type );

	// For explicitly setting activation parameters. 'Param' must be a 'CParam' struct from the correspondent layer
	template<class Param>
	CActivationDesc( TActivationFunction type, const Param& param );

	// Changing/setting parameters of the selected activation.
	// 'Param' must be a 'CParam' struct from the correspondent layer.
	template<class Param>
	void SetParam( const Param& param );

	/// etc.
};
```

## Layer types

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
	AF_Exp,
	AF_Log,

	AF_Count
};
```

TActivationFunction constant | Class name | Activation function
----------|-----------|--------------------
`AF_Linear` | [CLinearLayer](LinearLayer.md) | a linear activation function: `ax + b`
`AF_ELU` | [CELULayer](ELULayer.md) | `ELU` activation function
`AF_ReLU` | [CReLULayer](ReLULayer.md) | `ReLU` activation function
`AF_LeakyReLU` | [CLeakyReLULayer](LeakyReLULayer.md) | `LeakyReLU` activation function
`AF_Abs` | [CAbsLayer](AbsLayer.md) | `abs(x)` activation function
`AF_Sigmoid` | [CSigmoidLayer](SigmoidLayer.md) | `sigmoid` activation function
`AF_Tanh` | [CTanhLayer](TanhLayer.md) | `tanh` activation function
`AF_HardTanh` | [CHardTanhLayer](HardTanhLayer.md) | `HardTanh` activation function
`AF_HardSigmoid` | [CHardSigmoidLayer](HardSigmoidLayer.md) | `HardSigmoid` activation function
`AF_Power` | [CPowerLayer](PowerLayer.md) | `pow(x, exp)` activation function
`AF_HSwish` | [CHSwishLayer](HSwishLayer.md) | `h-swish` activation function
`AF_GELU` | [CGELULayer](GELULayer.md) | `x * F( X < x )` activation function, where X ~ N(0, 1)
`AF_Exp` | [CExpLayer](ExpLayer.md) | `exp` activation function
`AF_Log` | [CLogLayer](LogLayer.md) | `log` activation function
`AF_Erf` | [CErfLayer](ErfLayer.md) | `erf` activation function
`AF_Count` | | This is an auxiliary constant: it contains the number of supported activation functions.
