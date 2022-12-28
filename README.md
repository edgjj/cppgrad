# cppgrad

Maybe worst possible DL framework.

## Possibilities
- N-dim Tensor with mem. leaks and data corruption
- pytorch/tinygrad-like autograd engine
- CPU/CUDA backends
- Typical DL ops
- MPI COMM_WORLD distributed possibilities
- SGD optimizer (and distributed one too!)
- Cool itertools (maybe the best thing there)
- Little bit more

## Disclaimer
This was written as a research how modern n-dim arrays & reverse mode
dynamic graph automatic differentiation works.

Whole repository code smells like death, especially executors part, MPI one,
and some parts of Tensor.

Measured NN perf is around 1/5000 PyTorch on CPU, and 1/200 on GPU.

<br />

While isolated ops performance could be good (like dot product & matmul, see bench_dot & bench_matmul in examples), complete performance is pure trash, 
especially autograd.

In future these issues could be resolved, through complete remake of some parts (mostly Tensor & Autograd-related stuff).

But future is not yet now...

## Example snippet
```
    using namespace cppgrad;

    nn::Linear lin(16, 10, f64);
    nn::optim::SGD optim(lin, 1e-2);

    auto x = Tensor::full({ 1, 16 }, 0.5, f64);
    auto y = Tensor { { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };

	for (int i = 0; i < 384; i++) {
        auto out = lin(x)[0];
        auto loss = nn::mse_loss(out, y);

        optim.zero_grad();
        loss.backward();
        optim.step();
    }

    auto out = lin(x)[0];
    auto loss = nn::mse_loss(out, y);
```

## Quickstart
```
	$ git clone https://somegitshit.com/edgjj/cppgrad.git
	$ cd cppgrad && mkdir build
	$ cmake .. -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
	$ cmake --build . --config Release
	$ ctest .
```

## CMake options

| Flag             | Behavior                   | Default Value |
| ---------------- | -------------------------- | ------------- |
| `BUILD_EXAMPLES` | Build examples             | OFF           |
| `BUILD_TESTS`    | Build tests                | OFF           |
| `CPPGRAD_CUDA`   | Build CUDA backend         | ON            |
| `CPPGRAD_MPI`    | Build distributed features | OFF           |

## Hall of Fame

* **[micrograd](https://github.com/karpathy/micrograd)** - the reason why this exists
* **[tinygrad](https://github.com/geohot/tinygrad)** - the reason why this works
* **[PyTorch](https://github.com/pytorch/pytorch)** - grad_fn intensifies
* **[Andrew Karpathy](https://github.com/karpathy)** - a cool guy