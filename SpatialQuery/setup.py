from setuptools import setup, find_packages
# from setuptools import Extension
# import pybind11

# cpp_extension = Extension(
#     name="SpatialQueryEliasFanoDB",  # Full path to extension
#     sources=[
#         "SpatialQuery/scfind4sp/cpp_src/eliasFano.cpp",
#         "SpatialQuery/scfind4sp/cpp_src/QueryScore.cpp",
#         "SpatialQuery/scfind4sp/cpp_src/fp_growth.cpp",
#         "SpatialQuery/scfind4sp/cpp_src/serialization.cpp",
#         "SpatialQuery/scfind4sp/cpp_src/utils.cpp",
#     ],
#     include_dirs=["SpatialQuery/scfind4sp/cpp_src"] + [pybind11.get_include()],
#     language="c++",
#     extra_compile_args=["-std=c++11"],
# )

setup(
    name='SpatialQuery',
    version='0.1.4',
    packages=find_packages(),
    url='https://github.com/ShaokunAn/Spatial-Query',
    license='MIT',
    author='Shaokun An',
    author_email='shan12@bwh.harvard.edu',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.8',
    install_requires=[
        'setuptools>=68.0.0',
        'anndata>=0.8.0',
        'pandas>=2.0.3',
        'scipy',
        'matplotlib>=3.7.5',
        'mlxtend>=0.23.1',
        'seaborn>=0.13.2',
        'scikit-learn>=1.3.2',
        'statsmodels>=0.14.0',
    ],
    # ext_modules=[cpp_extension],
)
