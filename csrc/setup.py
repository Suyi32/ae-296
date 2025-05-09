from setuptools import setup
from torch.utils import cpp_extension

setup(name='tensor_cp',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'tensor_cp', 
            ["interface.cpp"],
            extra_compile_args=['-std=c++17'],
            libraries=['rt', 'pthread'],
        )
    ],
    package_data={'tensor_cp': ['tensor_cp.pyi']},
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
