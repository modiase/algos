project('c_project', 'c',
  version: '0.1.0',
  default_options: [
    'warning_level=3',
    'werror=false'
  ])

cc = meson.get_compiler('c')

c_flags = []

if cc.get_id() == 'gcc'
  c_flags += [
    '-Wall',
    '-Wextra',
    '-Wpedantic',
    '-Wformat=2',
    '-Wno-unused-parameter',
    '-Wshadow',
    '-Wwrite-strings',
    '-Wstrict-prototypes',
    '-Wold-style-definition',
    '-Wredundant-decls',
    '-Wnested-externs',
    '-Wmissing-include-dirs',
    '-Wjump-misses-init'
  ]
elif cc.get_id() == 'clang'
  c_flags += [
    '-Wall',
    '-Wextra',
    '-Wpedantic',
    '-Wformat=2',
    '-Wno-unused-parameter',
    '-Wshadow',
    '-Wwrite-strings',
    '-Wstrict-prototypes',
    '-Wold-style-definition',
    '-Wredundant-decls',
    '-Wnested-externs',
    '-Wmissing-include-dirs',
    '-Wconditional-uninitialized'
  ]
else
  error('Unsupported compiler: ' + cc.get_id() + '. Only GCC and Clang are supported.')
endif

add_project_arguments(c_flags, language: 'c')

inc_directory = include_directories('inc')

binomial_heap_lib = shared_library('binomial_heap',
  'src/binomial-heap.c',
  include_directories: inc_directory,
  install: false,
  version: meson.project_version(),
  soversion: '0',
  c_args: c_flags)

binomial_heap_dep = declare_dependency(
  link_with: binomial_heap_lib,
  include_directories: inc_directory)

main_exe = executable('main',
  'src/main.c',
  dependencies: binomial_heap_dep,
  install: false)

test_exe = executable('test_main',
  'test/test_main.c',
  dependencies: binomial_heap_dep,
  install: false)

test('basic_test', test_exe)
