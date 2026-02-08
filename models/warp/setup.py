from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

# Automatically find source files in the subdirectory
# We look for all .cpp and .cu files in Forward_Warp/cuda/
sources = glob.glob(os.path.join('Forward_Warp', 'cuda', '*.cpp')) + \
          glob.glob(os.path.join('Forward_Warp', 'cuda', '*.cu'))

setup(
    name='Forward_Warp',
    version='0.0.1',
    author="lizhihao6",
    author_email="lizhihao6@outlook.com",
    # Include the Python package
    packages=find_packages(), 
    ext_modules=[
        CUDAExtension(
            name='forward_warp_cuda', # The name imported in your python code
            sources=sources,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
