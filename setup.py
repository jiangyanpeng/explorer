from setuptools import setup, find_packages

version_file = 'modelexplorer/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def get_long():
    long = ''
    with open('README.md', 'r') as f:
        long = f.read()
    return long


setup(name='explorer',
      version=get_version(),
      keywords=['deep learning', 'script helper', 'tools'],
      description='''
      Explorer for ONNX/Caffe/ncnn.
      ''',
      long_description=get_long(),
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

      author="Jiang Yanpeng",
      author_email="jiangyanpeng@sensetime.com",
      url='https://github.com/jiangyanpeng/explorer.git',
      platforms='any',
      install_requires=['colorama', 'requests', 'numpy', 'future', 'onnx', 'rich',
                        'deprecated', 'alfred-py', 'onnxruntime', 'tabulate']
      )
