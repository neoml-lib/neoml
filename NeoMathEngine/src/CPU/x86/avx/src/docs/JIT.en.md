# NeoMathEngineAvx Module

## Introduction
The NeoMathEngineAvx module is designed to accelerate certain NeoML methods by implementing them using the AVX/AVX2/FMA instruction sets.
This allows the code to run on relatively older x86_64 processors (Haswell and newer) as well as AMD processors that support the corresponding instruction sets.

Currently, the module implements 3 main subsystems: direct convolution, matrix multiplication kernels, and a set of various primitives used in neural networks.

The matrix multiplication kernels are written using intrinsic calls for use in the `CMatrixMultiplier` class.
According to benchmarks, matrix multiplication with these kernels performs better than MKL on AMD processors but generally lags behind MKL on Intel processors, even with the same AVX/AVX2/FMA instruction set.

Direct convolution and primitives are implemented using the OpenSource library [xbyak](https://github.com/herumi/xbyak).
This library enables the generation of machine instructions at runtime, so we will refer to this technology as the **JIT compiler** or simply **JIT**.
One of the notable features of `xbyak` is its straightforward API, which allows writing code that closely resembles Intel assembly syntax.
The main advantages of the JIT compiler are the minimization of conditional jumps, the calculation of all offsets, and the substitution of constant values at the JIT code compilation stage, as well as loop unrolling for better pipeline code prefetching.

## Module Descriptions

### Matrix Multiplication Kernels
This part is pure C++ and does not need further description in this document.

### Forward Convolution
Originally, this module was written using intrinsic calls but was later transitioned to JIT.
However, the implementation as a template class remained, although it is no longer necessary because template parameters do not affect the JIT code generation time, and templates are no longer needed after code generation.
*In the future, it would be beneficial to remove template parameters from the class to simplify the code.*

JIT code for **Forward Convolution** is generated separately for each descriptor, as it depends on various descriptor parameters.
This is one of those cases where acceleration is achieved partly through the use of pre-known constants and offsets in the instructions.
The functionality of this module was previously described for intrinsic implementations, and the JIT version has not fundamentally changed the core logic.

The workings of JIT will be discussed in the next section.

### Primitives
JIT is a very unconventional way of writing code: on one hand, you write the logic of the JIT compiler in C++, benefiting from all the advantages and new features of the language;
on the other hand, you write code that is similar to assembly language, resulting in compact and relatively optimized code (though not always readable for an unprepared programmer).
Most importantly, you write code knowing the conditions under which it will run: how many times, with which variables/constants, etc., thus enabling you to achieve maximum performance.
These 3 aspects help in significantly improving performance.

### Preparation and Reference Materials
1. Start by familiarizing yourself with the basics of Assembler language and Intel syntax.
2. Read the documentation on [xbyak](https://github.com/herumi/xbyak); it is quite concise and well-structured with examples.
3. For documentation on assembly instructions, I mainly used the following sites:
   * [x86 and amd64 instruction reference](https://www.felixcloutier.com/x86/index.html)
   * [Intel® Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
   * [X86-64_Instruction_Encoding](https://wiki.osdev.org/X86-64_Instruction_Encoding) - useful for understanding addressing modes in command formation.

That should cover everything needed for confident work with `xbyak`.

### Issues or peculiarities encountered when using xbyak
I believe this section should be placed BEFORE the description of working with `xbyak`.

1. **Mixing AVX and SSE Instructions:**
   During execution, mixing AVX and SSE instructions can cause significant performance degradation due to potential reloading of the SIMD instruction module.
   More details on this can be found in the [Intel® 64 and IA - 32 Architectures Optimization Reference Manual](https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf),
   Section 11.3 **MIXING AVX CODE WITH SSE CODE**: *Assembly/Compiler Coding Rule 72. (H impact, H generality) Add the VZEROUPPER instruction after 256-bit AVX instructions are executed and before any function call that might execute SSE code.*
   *Add VZEROUPPER at the end of any function that uses 256-bit AVX instructions.*

2. **Error Checking in xbyak:**
   `xbyak` has the beneficial feature of making it very difficult to form incorrect instructions. Inside function calls, there are numerous checks to prevent the use of incorrect operands in instructions.

3. **Challenges with Prefetch Instructions:**
   Gaining performance improvements from using prefetch instructions is challenging.
   This is because modern processors have very effective hardware implementations for memory access pattern recognition and automatically prefetch necessary blocks.

### Description of the `CPrimitivesJit` class

#### General Information
The class interface implements 2 mechanisms for calling primitive functions:
1. Directly invoking the method (JIT implementation of the LSTM network),
2. Obtaining a pointer to a specific primitive (allows for seamless replacement of primitive calls in the `CCpuMathEngine` class).

The greatest performance gains can be achieved for computation-intensive operations.
For this reason, simple primitives often show unstable performance compared to SSE intrinsic implementations, as memory access time, particularly cache misses, can cause significant performance variability.

### Container `gens` holding JIT code for primitive functions
All implemented JIT functions are listed in the `TPrimitive` enum.
The class defines a `CGenerator` structure, which essentially is an instance of the `Xbyak::CodeGenerator` class responsible for generating JIT code.
The **`gens`** array stores all generated JIT primitives, and their generation occurs upon the first access to a particular primitive in the `GetFunctionRawPtr()` method.

### Constants Table
The constants table is defined using **`TTableKey`** keys and two containers: **`table`** and **`tableOffsets`**.
This table allows for centralized storage of all constants needed for executing JIT code and provides easy access to them.
Since it is not possible to directly set the initializing value of `ymm` registers in instruction code, they must be loaded from memory.
Centralizing all constants in one place benefits cache performance.

The constants table is initialized once in the `CPrimitivesJit` constructor by calling the `initTable` method.
In each JIT function, the address of the constants table (if used) is located in the `regTablePtr` register (**`r10`**).
The address for initializing `ymm` registers can be obtained using the `getOfft` or `getAddr` methods, passing the necessary offsets.
The `getAddr` method returns the direct address descriptor `Xbyak::Address`, which implements relative addressing (relative to the `regTablePtr` register).

### Structure of Primitive Functions
JIT primitives are created in the template method `initPrimitive`, which
* either directly contains the JIT function generation code
* or serves as a wrapper for another template method with the name `init...`, which generalizes the implementation of similar functions (typically basic mathematical primitives).

Nevertheless, the following essential steps can be highlighted:
* **Creating an instance of the `CGenerator` class**:
  - This is used for generating JIT code.
* **Defining registers that need to be preserved before executing the function `preservedReg64` and `preservedYmm`**:
  - For Windows and Unix, these are different sets of registers; more details can be found in the calling conventions for the specific OS.
* **Calling the mandatory `Prologue` method**:
  - This method adds the necessary prologue for any function, saves the required registers on the stack, and calculates the address descriptor pointing to the stack area containing the function arguments (if any).
* **Defining registers used inside the function and their initialization**:
  - Initialization either from registers or from stack values, it is also important to understand the calling convention here.
    For example, Windows passes only 4 arguments through registers, while Unix passes 6.
    Additionally, if arguments are multiplied by floating-point values, they would stored in different `xmm` registers in Windows and Unix.
    If there are more function arguments than the allocated registers, the remaining arguments are passed via the stack.
    Windows also has a `ShadowSpace` concept: 32-byte reserved area on the stack before the arguments.
* **Implementing a lambda function**:
  - This lambda performs the core computational operations for the primitive.
* **Batch processing input data with loop unrolling:**
  - The lambda function from the previous step is applied to it.
    The number of elements that can be unrolled in one loop iteration is determined by the number of `ymm` registers involved and the number of elements in typical use cases.
    For complex primitives like `Exp`, `Sigmoid`, `Tanh`, and `RestOfLstm`, which use many `ymm` registers, no more than 2 values are processed at a time.
* **Handling tail of input data**:
  - If the length of the input data is not a multiple of 8, tail values are processed separately.
    Tail handling differs only in that data reading and writing are performed with a mask.
* **Calling the mandatory `Epilogue` method**:
  - This restores previously saved registers, executes the required `VZEROUPPER` instruction (as mentioned earlier), and restores the frame and stack pointers (using the `leave` instruction).

For simpler primitives, there is not much additional detail.
The batch processing code for varying numbers of input data is unified and implemented in the `insertSimpleMathFunction()` function.

### Description of Complex Primitives
Complex primitives consist of other primitives (**Sigmoid** and **RestOfLstm**) or are computed using polynomials (**Tanh** and **Exp**).
These primitives use many `ymm` registers to process a block of 8 input values, typically unrolling no more than 2 such blocks at a time.

In the `insertPrimitive` function, which processes 2 blocks (2 input `ymm`), we would have to write each line twice, and have another similar function that processes only 1 block at a time.
Such an approach would inevitably lead to code divergence and hard-to-detect bugs, so in these primitives we operate not with individual `ymm`, but with the `ymmVec_t` vector.
During initialization we either manually check how many blocks should be processed:
```
ymmVec_t forget = wholeYmmNumber == 2 ? ymmVec_t{ ymm0, ymm1 } : ymmVec_t{ ymm0 };
```
or use a special function `initFromAux()`, which slices the auxiliary register vector `ymmAux` into small blocks, each of which matches the size of the input data (`ymmSrc`).

Such manipulation of `ymm` vectors required us to overload various methods of the `Xbyak::CodeGenerator` class responsible for generating instructions.
These overloaded functions are defined in the base class `CJitCommon`.
To enable single-line function overloading, we had to define several macros and template functions for different numbers and combinations of arguments.
As a result, we achieved a syntax similar to what `xbyak` offers for working with register vectors.

#### Tanh
The computation of the hyperbolic tangent (`tanh`) function is described in detail through comments.

Segments based the Argument's range:
1. **Linear Segment:**
   - `[0; linear_ubound]`, where `tanh(x) = x`.

2. **Polynomial Segments:**
   - `[linear_ubound; 0x1.8p-12]` is the first half of a binom.
   - `[0x1.8p-12; 0x1.0p-11], ..., [0x1.8p2; 0x1.0p3]` are 29 half-binoms.
   - `[0x1.0p3; saturation_ubound]`
   - This gives a total of 31 intervals where `tanh` is calculated using a 6th-degree polynomial.

3. **Saturation Segment:**
   - `[0x1.205966p3; saturation_ubound]`, where `tanh(x) = 1`.

The other steps in code for computing `tanh(x)` are:
1. **Discard the sign:** 
   - The absolute value of the argument is used for the computation, and the sign is reintroduced at the end.

2. **Determine the polynomial segment:**
   - Identify which polynomial segment the argument falls into.

3. **Compute `tanh` using a polynomial:**
   - For polynomial segment `tanh(x)` is computed using a polynomial approximation.

4. **Handle Linear and Saturation Segments:**
   - For the linear segment, simply use `x`, and for the saturation segment, use `1`.

5. **Return the sign:**
   - Add the sign back to the computed result.

#### Exp
The exponential function (`exp`) is computed using a polynomial approximation.
To calculate `exp(x)`, the argument is split into 2 parts based on `ln(2)`:

1. **Decompose the Argument:**
   - The result of `x / ln(2)` is split into its integer part `n` and the remainder `r`:
     ```
     x = n * ln(2) + r
     exp(x) = exp(n * ln(2) + r) = exp(ln(2))^n * exp(r) = 2^n * exp(r)
     ```
   - Hence, `2^n` is calculated using a shift operation, and `exp(r)` is approximated using a polynomial on the interval `[0; ln(2)]`.

2. **Limit Argument to Avoid Overflow:**
   - Before calculating the exponent, limit the argument from above and below to avoid overflow.
     Constants `ExpFltMax` and `ExpFltMin` are specified in hex format. If `ExpFltMax` is increased by one, the result will be `inf` on overflow, which does not agree with the similar result in MKL.

**Function Operations:**
1. Save a mask for a value less than `ExpFltMin`, and at the end of the function use it to set corresponding results to zero:
   ```
   gen.vcmpltps(ymmMask, ymmSrc, getAddr(TTableKey::ExpFltMin));
   ```
2. Limit the argument `x` from above and below and cache this value to use further
   ```
   gen.vminps(ymmSrc, ymmSrc, getAddr(TTableKey::ExpFltMax));
   gen.vmaxps(ymmSrc, ymmSrc, getAddr(TTableKey::ExpFltMin));
   gen.vmovups(ymmAux1, ymmSrc);
   ```
3. Calculate the integer part of the division `x / ln(2)`, adding `0.5` to shift the remainder of the division to the zero region for a more accurate calculation of the polynomial.
   ```
   n = round( x / ln(2) + 0.5 ) = round( x * log2(e) + 0.5 )
   ```
4. Calculate the remainder `r = x - n * ln(2)`
5. In case `n == 128`, to avoid an overflow when calculating `2^n`, so need to do the following:
   ```
   exp(x) = 2^n * exp(r) = 2^(n-1) * exp(r) * 2
   ```
   the expression `2^(n-1) * exp(r)` will be less than `2^127` and as a result there will be no overflow.

   Calculate `2^(n-1)` using the shift operation, adding `ExpBias` to maintain the floating point format:
   ```
   gen.vsubps( ymmSrc, ymmSrc, getAddr( TTableKey::One ) );
   gen.vcvtps2dq( ymmAux2, ymmSrc );
   gen.vpaddd( ymmAux2, ymmAux2, getAddr( TTableKey::ExpBias ) );
   gen.vpslld( ymmAux2, ymmAux2, MantissaNumBits );
   ```
6. Apply the mask obtained in step 1, zeroing out the values that are less than `ExpFltMin`
   ```
   gen.vxorps( ymmSrc, ymmSrc, ymmSrc );
   gen.vblendvps( ymmAux2, ymmAux2, ymmSrc, ymmMask );
   ```
7. Calculate `exp(r)` using a polynomial
8. Multiply the obtained value first by `2^(n-1)` then by `2` to avoid the problem obtained in step 5.

#### Sigmoid
The Sigmoid function is computed using the exponential function as follows:

$\[ \text{sigmoid}(x) = \frac{\exp(x)}{1 + \exp(x)} \]$

**In the Method `insertPrimitive<Sigmoid>`:**

- **Initialization of Registers:** 
  - The final register `ymmAux` is pre-initialized to ones.
    This is because `insertPrimitive<Sigmoid>` is also used in `insertPrimitive<RestOfLstm>`, and pre-initializing `ymmAux` simplifies the logic and avoids redundant operations.
  
- **Functional Object `afterPrologue`:**
  - To maintain the unified method approach in `initActivationFunction()`, a parameter `afterPrologue` is added.
    This parameter is a functional object that is invoked after the prologue to initialize the `ymmAux` register with ones for the sigmoid function.

#### RestOfLstm
The implementation of the `RestOfLstm` primitive might appear complex, but it can be better understood with the aid of a diagram [RestOfLstm.pdf](https://github.com/neoml-lib/neoml/NeoMathEngine/src/CPU/x86/avx/src/docs/RestOfLstm.pdf).

![image](https://github.com/user-attachments/assets/d64e8bbf-e023-40f0-ad27-7cedca41db4b)

This diagram represents the classic LSTM cell and its implementation, broken down into 3 blocks.
Each block is designed to use the maximum number of `ymm` registers (up to 16) while avoiding excessive register usage.

Each block is labeled with the number of `ymm` registers used *at the input*, *maximum within* the block, and *at the output*.

The main executions are performed in the lambda function `insertCode()`, which processes 1 or 2 `ymm` input registers at a time.

The LSTM layer is recurrent, with an outer loop for processing 1 LSTM step and inner loops for handling data in chunks (first 16 values, then 8, and finally the remainder).
```cpp
	// *** Main loop ***
	gen.StartDownCountLoop(regObjectsCount, 1);

	// ... Inner processing of 1 LSTM step

	// *** Stop Main loop ***
	gen.StopDownCountLoop();
```

After each inner loop, pointers are nullified.
Methods `StartDownCountLoop()` and `StopDownCountLoop()` are designed to handle nested loops by managing labels for transitions in the stack and tracking the nesting level.

## Conclusion
The JIT primitives were designed to be scalable  and easy to extend by new primitives. However, the code could benefit from refactoring:
1. Consider limiting access to primitives solely through pointer retrieval and removing multi-threading support if not needed.
2. The method `insertSimpleMathFunction()` standardizes loop unrolling for simple mathematical primitives, could be extended to handle complex primitives similarly.
