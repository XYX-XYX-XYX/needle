.PHONY: lib pybind clean format all

PYTHON := /root/miniconda3/envs/needle/bin/python3

all: lib

lib:
	@mkdir -p build
	@cd build; cmake .. \
		-DNEEDLE_USE_FLASHATTN_STUB=ON \
		-DPython_EXECUTABLE=$(PYTHON) \
		-DPython3_EXECUTABLE=$(PYTHON) \
		-DPYTHON_EXECUTABLE=$(PYTHON)
	@cd build; $(MAKE)

format:
	$(PYTHON) -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so