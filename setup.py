# -*- coding: utf-8 -*-
# file: setup.py
# author: JinTian
# time: 04/02/2019 12:16 PM
# Copyright 2019 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""
install onnxexp into local bin dir.
"""
from setuptools import setup, find_packages

version_file = 'modelexplorer/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


setup(name='modelexplorer',
      version=get_version(),
      keywords=['deep learning', 'script helper', 'tools'],
      description='''
      Explorer for ONNX.
      ''',
      long_description='''
      onnxexp provides easy way to explore model structure and node detail in onnx model.
      ''',
      license='Apache 2.0',
      packages=[
          'modelexplorer',
          'modelexplorer.proto',
      ],
      entry_points={
          'console_scripts': [
              'onnxexp = modelexplorer.modelexplorer:main'
          ]
      },

      author="Lucas Jin",
      author_email="jinfagang19@163.com",
      url='https://github.com/jinfagang/modelexplorer',
      platforms='any',
      install_requires=['colorama', 'requests', 'numpy', 'future', 'onnx', 'rich',
                        'deprecated', 'alfred-py', 'onnxruntime', 'tabulate']
      )