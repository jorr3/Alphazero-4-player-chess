import pybind11
import shutil

from setuptools import setup
from glob import glob
from torch.utils import cpp_extension


class AlphazeroCppBuildExt(cpp_extension.BuildExtension):
    def build_extension(self, ext):
        super().build_extension(ext)
        # shutil.rmtree("./build")


alphazero_cpp_source_files = glob("./src/cpp/engine/*.cpp") + glob("./src/cpp/*.cpp")
libs_path = "./src/cpp/libs"

ext_modules = [
    cpp_extension.CppExtension(
        name="alphazero_cpp",
        sources=alphazero_cpp_source_files,
        include_dirs=[
            "./src/cpp",
            pybind11.get_include(),
        ]
        + cpp_extension.include_paths(),
        library_dirs=[libs_path + "/libtorch/lib"],
        libraries=["torch"],
        cxx_std=17,
        extra_link_args=["/DEBUG", "/PDB:alphazero_cpp.pdb"],
        extra_compile_args=["/Zi", "/Od", "-g"],
    )
]

setup(
    name="alphazero_cpp",
    # version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": AlphazeroCppBuildExt},
)

# build command: python setup.py build_ext --inplace
