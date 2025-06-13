from setuptools import Extension, setup

ring_buffer_extension = Extension(
    "ring_buffer",
    sources=["src/ring_buffer_module.c", "src/buffer.c"],
    include_dirs=["src"],
    extra_compile_args=["-O3", "-std=c99"],
)

setup(ext_modules=[ring_buffer_extension])
