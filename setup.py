import pybind11
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from glob import glob
import shutil
import os

class CustomBuildExt(build_ext):
    def build_extension(self, ext):
        super().build_extension(ext)

        # Define the target directory for the moved files
        target_dir = os.path.abspath('./venv')
        os.makedirs(target_dir, exist_ok=True)

        # Determine the original and new file paths
        original_path = self.get_ext_fullpath(ext.name)
        new_file_name = ext.name.split('.')[0] + os.path.splitext(original_path)[-1]
        new_path = os.path.join(target_dir, new_file_name)

        # Move the file, then explicitly check and remove the original if it still exists
        shutil.move(original_path, new_path)
        print(f"Moved {original_path} to {new_path}")

        # Check if the original file still exists for some reason and remove it explicitly
        if os.path.exists(original_path):
            os.remove(original_path)
            print(f"Explicitly removed the original file: {original_path}")


source_files = sorted(glob("./engine/*.cc"))

ext_modules = [
    Pybind11Extension(
        'algos',
        sources=['./src/cpp_algorithm_implementations/algos.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['/std:c++latest']
    ),
    Pybind11Extension(
        "chessenv",
        source_files,
        include_dirs=["./engine"],
        cxx_std=17,
        extra_compile_args=['-g', '/Zi'],
        extra_link_args=['-g', '/DEBUG', '/PDB:chessenv.pdb'],
    ),
]

setup(
    name="ChessEngine",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
)


# build command: python setup.py build_ext --inplace
