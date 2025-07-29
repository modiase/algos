project('fibonacci-heap', 'c',
  version : '1.0.0',
  default_options : ['warning_level=3', 'c_std=c99'])

# Compiler flags
cc = meson.get_compiler('c')
add_project_arguments('-Wall', '-Wextra', '-Wpedantic', language: 'c')

# Source files
src_files = [
  'src/heap.c',
  'src/utils.h',
  'src/heap.h'
]

# Create the library
fib_heap_lib = static_library('fib_heap',
  src_files,
  include_directories: include_directories('src'),
  install: false)

# Create the test executable
test_exe = executable('test_heap',
  'tests/test_heap.c',
  link_with: fib_heap_lib,
  include_directories: include_directories('src'),
  install: false)

# Run tests
test('fibonacci-heap-tests', test_exe)

# Install targets (optional)
install_headers('src/heap.h', 'src/utils.h') 