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

main_exe = executable('main',
  'src/main.c',
  include_directories: inc_directory,
  install: false)

test_basic_operations_exe = executable('test_basic_operations',
  'test/test_basic_operations.c',
  include_directories: inc_directory,
  install: false)

test_union_operations_exe = executable('test_union_operations',
  'test/test_union_operations.c',
  include_directories: inc_directory,
  install: false)

test_complex_scenarios_exe = executable('test_complex_scenarios',
  'test/test_complex_scenarios.c',
  include_directories: inc_directory,
  install: false)

test_exe = executable('test_main',
  'test/test_main.c',
  include_directories: inc_directory,
  install: false)

test('basic_operations', test_basic_operations_exe)
test('union_operations', test_union_operations_exe)
test('complex_scenarios', test_complex_scenarios_exe)
test('basic_test', test_exe)
