cmake_minimum_required(VERSION 3.5)
project(needle C CXX)
cmake_policy(SET CMP0146 OLD)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NEEDLE_USE_FLASHATTN_STUB "Build FlashAttention stub hooks into the CUDA backend" ON)
option(NEEDLE_CUDA_LINEINFO "Compile CUDA backend with source line info for profiling" ON)
set(NEEDLE_CUDA_MAIN_ARCH "80" CACHE STRING "CUDA SM architecture for the pybind CUDA backend TU")
set(NEEDLE_FLASHATTN_SM80_ARCH "80" CACHE STRING "CUDA SM architecture for src/flash_attention_sm80.cu")
set(NEEDLE_FLASHATTN_SM90_ARCH "90a" CACHE STRING "CUDA SM architecture for src/flash_attention_sm90.cu")
set(NEEDLE_CUTLASS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass" CACHE PATH "Path to the CUTLASS checkout")

# find correct version of Python
execute_process(COMMAND python3-config --prefix
  OUTPUT_VARIABLE Python_ROOT_DIR)
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})


if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
else()
  set(CMAKE_CXX_FLAGS "-O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
### CPU BACKEND ###
###################
add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc)
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})
pybind11_extension(ndarray_backend_cpu)
pybind11_strip(ndarray_backend_cpu)


# directly output to ffi folder
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
  CXX_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()



####################
### CUDA BACKEND ###
####################
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

  set(NEEDLE_ENABLE_FLASHATTN_STUB OFF)
  if(NEEDLE_USE_FLASHATTN_STUB)
    if(EXISTS "${NEEDLE_CUTLASS_PATH}/include/cutlass/cutlass.h")
      set(NEEDLE_ENABLE_FLASHATTN_STUB ON)
      message(STATUS "Building CUDA backend with FlashAttention stub support")
    else()
      message(WARNING "FlashAttention stub requested but CUTLASS was not found at ${NEEDLE_CUTLASS_PATH}; building CUDA backend without flash stub support")
    endif()
  endif()

  function(needle_cuda_arch_flags out_var arch)
    set(${out_var} "-gencode=arch=compute_${arch},code=sm_${arch}" PARENT_SCOPE)
  endfunction()

  set(NEEDLE_CUDA_BASE_OPTIONS
    -O3
    --use_fast_math
    -Xptxas -O3
    -Xptxas -v
    -Xptxas --warn-on-spills
  )
  if(NEEDLE_CUDA_LINEINFO)
    list(APPEND NEEDLE_CUDA_BASE_OPTIONS -lineinfo)
    message(STATUS "Compiling CUDA backend with -lineinfo")
  endif()

  needle_cuda_arch_flags(NEEDLE_CUDA_MAIN_ARCH_FLAGS ${NEEDLE_CUDA_MAIN_ARCH})
  set(NEEDLE_CUDA_OPTIONS ${NEEDLE_CUDA_MAIN_ARCH_FLAGS} ${NEEDLE_CUDA_BASE_OPTIONS})
  message(STATUS "Compiling CUDA pybind backend TU for SM ${NEEDLE_CUDA_MAIN_ARCH}")

  set(NEEDLE_FLASHATTN_OBJECTS)
  if(NEEDLE_ENABLE_FLASHATTN_STUB)
    needle_cuda_arch_flags(NEEDLE_FLASHATTN_SM80_ARCH_FLAGS ${NEEDLE_FLASHATTN_SM80_ARCH})
    needle_cuda_arch_flags(NEEDLE_FLASHATTN_SM90_ARCH_FLAGS ${NEEDLE_FLASHATTN_SM90_ARCH})
    set(NEEDLE_FLASHATTN_COMMON_OPTIONS
      ${NEEDLE_CUDA_BASE_OPTIONS}
      -Xcompiler -fPIC
      --expt-relaxed-constexpr
      -DNEEDLE_ENABLE_FLASHATTN_STUB=1
      "-I${NEEDLE_CUTLASS_PATH}/include"
    )

    message(STATUS "Compiling FlashAttention SM80 TU for SM ${NEEDLE_FLASHATTN_SM80_ARCH}")
    CUDA_COMPILE(NEEDLE_FLASHATTN_SM80_OBJECTS
      src/flash_attention_sm80.cu
      OPTIONS ${NEEDLE_FLASHATTN_COMMON_OPTIONS} ${NEEDLE_FLASHATTN_SM80_ARCH_FLAGS}
    )

    message(STATUS "Compiling FlashAttention SM90 TU for SM ${NEEDLE_FLASHATTN_SM90_ARCH}")
    CUDA_COMPILE(NEEDLE_FLASHATTN_SM90_OBJECTS
      src/flash_attention_sm90.cu
      OPTIONS ${NEEDLE_FLASHATTN_COMMON_OPTIONS} ${NEEDLE_FLASHATTN_SM90_ARCH_FLAGS}
    )

    list(APPEND NEEDLE_FLASHATTN_OBJECTS ${NEEDLE_FLASHATTN_SM80_OBJECTS} ${NEEDLE_FLASHATTN_SM90_OBJECTS})
  endif()

  # set arch flags properly
  CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE
    src/ndarray_backend_cuda.cu
    ${NEEDLE_FLASHATTN_OBJECTS}
    OPTIONS ${NEEDLE_CUDA_OPTIONS}
  )

  if(NEEDLE_ENABLE_FLASHATTN_STUB)
    target_compile_definitions(ndarray_backend_cuda PRIVATE NEEDLE_ENABLE_FLASHATTN_STUB=1)
    target_include_directories(ndarray_backend_cuda SYSTEM PRIVATE
      ${NEEDLE_CUTLASS_PATH}/include
    )
  endif()

  target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
  pybind11_extension(ndarray_backend_cuda)
  pybind11_strip(ndarray_backend_cuda)

  # directly output to ffi folder
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)

endif()
