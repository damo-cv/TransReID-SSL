from setuptools import setup, find_packages


setup(name='TranReID',
      version='1.0.0',
      description='TransReID: Transformer-based Object Re-Identification',
      author='xxx',
      author_email='xxx',
      url='xxx',
      install_requires=[
          'numpy', 'torch==1.6.0', 'torchvision==0.7.0',
          'h5py', 'opencv-python', 'yacs', 'timm==0.3.2'
          ],
      packages=find_packages(),
      keywords=[
          'Pure Transformer',
          'Object Re-identification'
      ])
