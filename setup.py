from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

__version__ = "placeholder"
exec(open('vision_models_playground/metadata.py').read())

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

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
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
