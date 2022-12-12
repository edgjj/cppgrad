# cppgrad

A not so header-only C++ micrograd ripoff.

## Possibilities
- Same as micrograd
- Little bit more
- No dependencies except libstd

## Quickstart
```
	$ git clone https://somegitshit.com/edgjj/cppgrad.git
	$ cd cppgrad && mkdir build
	$ cmake .. -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
	$ cmake --build . --config Release
	$ ctest .
```

## CMake options

| Flag             | Behavior       |
| ---------------- | -------------- |
| `BUILD_EXAMPLES` | Build examples |
| `BUILD_TESTS`    | Build tests    |

## Hall of Fame

* **[micrograd](https://github.com/karpathy/micrograd)** - the reason why this exists
* **[Andrew Karpathy](https://github.com/karpathy)** - a cool guy