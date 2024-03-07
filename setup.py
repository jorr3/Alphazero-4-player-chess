import pybind11
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from glob import glob
import shutil
import os


class CustomBuildExt(build_ext):
    def initialize_options(self):
        super().initialize_options()
        self.build_lib = './venv'

    def run(self):
        super().run()

        new_file_locations = {
            'vc140.pdb': 'venv/algos.pyd',
            'chessenv.cp311-win_amd64.pyd': 'venv/chessenv.pyd',
            "algos.cp311-win_amd64.pyd": "venv/algos.pyd"
        }

        for src_filename, dst_filename in new_file_locations.items():
            src_filepath = os.path.join(self.build_lib, src_filename)
            dst_filepath = os.path.join(self.build_lib, dst_filename)

            if os.path.exists(src_filepath):
                shutil.move(src_filepath, dst_filepath)
                os.remove(src_filename)


source_files = sorted(glob("./engine/*.cc"))

ext_modules = [
Pybind11Extension(
        'algos',
        sources=['./alphazero/cpp_algorithm_implementations/algos.cpp'],
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

# command: python setup.py build_ext --inplace
