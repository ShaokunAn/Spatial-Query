from setuptools import Extension
from pybind11.setup_helpers import build_ext
import pybind11

# C++ extension must be defined here because pyproject.toml does not support ext_modules
cpp_extension = Extension(
    name="SpatialQueryEliasFanoDB",
    sources=[
        "SpatialQuery/scfind4sp/cpp_src/eliasFano.cpp",
        "SpatialQuery/scfind4sp/cpp_src/QueryScore.cpp",
        "SpatialQuery/scfind4sp/cpp_src/fp_growth.cpp",
        "SpatialQuery/scfind4sp/cpp_src/serialization.cpp",
        "SpatialQuery/scfind4sp/cpp_src/utils.cpp",
    ],
    include_dirs=["SpatialQuery/scfind4sp/cpp_src"] + [pybind11.get_include()],
    language="c++",
    extra_compile_args=["-std=c++11"],
)

# Metadata lives in pyproject.toml; only ext_modules is specified here
from setuptools import setup
setup(ext_modules=[cpp_extension])
