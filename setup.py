#!/usr/bin/env python3

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='romanesco',
    version='0.2',
    description='A vanilla recurrent neural network (RNN) language model',
    url='http://github.com/zurichnlp/romanesco',
    author='Samuel Läubli, Mathias Müller',
    author_email='laeubli@cl.uzh.ch, mmueller@cl.uzh.ch',
    license='LGPL',
    packages=['romanesco'],
    scripts=['bin/romanesco'],
    install_requires = [
        'numpy',
        'tensorflow-gpu'
    ])
