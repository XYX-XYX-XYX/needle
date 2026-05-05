cmake_minimum_required(VERSION 3.5)
project(needle C CXX)
cmake_policy(SET CMP0146 OLD)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NEEDLE_USE_FLASHATTN_STUB "Build FlashAttention stub hooks into the CUDA backend" ON)
option(NEEDLE_CUDA_LINEINFO "Compile CUDA backend with source line info for profiling" ON)
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

  # invoke nvidia smi to detect if we really have a GPU
  execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
  if(NV_RET EQUAL "0")
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    # set to 5.0 the flag of K80
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 8.0)
  endif()

  set(NEEDLE_CUDA_OPTIONS ${ARCH_FLAGS})
  if(NEEDLE_CUDA_LINEINFO)
    list(APPEND NEEDLE_CUDA_OPTIONS -lineinfo)
    message(STATUS "Compiling CUDA backend with -lineinfo")
  endif()

  # set arch flags properly
  CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${NEEDLE_CUDA_OPTIONS})

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
