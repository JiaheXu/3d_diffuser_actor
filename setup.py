import os.path

from setuptools import setup


core_requirements = [
    "einops",
    "torch",
    "numpy",
    "torchvision",
    "diffusers",
    "dgl",
    "flash_attn",
]

setup(name='3docp',
      version='0.1',
      description='3D object centric planning',
      author='Jiahe Xu',
      author_email='xjh4438318846@gmail.com',
      url='https://JiaheXu.github.io/',
      install_requires=core_requirements,
      packages=[
            '3docp',
      ],
)
