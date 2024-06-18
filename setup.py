from setuptools import setup, find_packages

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
        'numpy>=1.24.4',
        'pandas>=2.0.3',
        'scipy'
    ],

)
