from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

__version__ = "placeholder"
exec(open('vision_models_playground/metadata.py').read())

setup(
    name='vision_models_playground',
    packages=find_packages(),
    version=__version__,
    license='MIT',
    description="Akriel's vision models playground",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Alexandru Stirbu',
    author_email='Stirbu.Alexandru.Net@outlook.com',
    url='https://github.com/Akrielz/vision_models_playground',
    keywords=[
        'artificial intelligence',
        'deep learning',
        'transformer',
        'attention mechanism',
        'computer vision',
    ],
    install_requires=[
        'einops>=0.4.1',
        'torch>=1.10.0',
        'rotary-embedding-torch>=0.1.0',
        'torchvision>=0.11.1',
        'colorama>=0.4.5',
        'tqdm>=4.64.0',
        'torchmetrics>=0.9.3',
        'numpy>=1.22.0',
        'matplotlib>=3.5.2',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
