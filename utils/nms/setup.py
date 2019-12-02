from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='r_nms',
    ext_modules=[
        CUDAExtension('r_nms', [
            'src/rotate_polygon_nms.cpp',
            'src/rotate_polygon_nms_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})

