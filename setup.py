from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

__version__ = "placeholder"
__classifiers__ = ["placeholder"]
__keywords__ = "placeholder"
__requirements__ = "placeholder"
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
    keywords=__keywords__,
    install_requires=__requirements__,
    classifiers=__classifiers__,
)
