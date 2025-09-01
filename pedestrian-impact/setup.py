# from setuptools import setup
from distutils.core import setup

from setuptools import find_packages

setup(name='pedestrian_impact',
      version='0.0.1',
      packages=find_packages(),  # ["pedestrian_impact/"],
      package_data={'': ['*.xml', '*.png', '*.stl', '*.p']},
      install_requires=['numpy',
                        # 'gymnasium-robotics[all]',
                        'gymnasium[all]',
                        'mujoco==2.3.6',
                        # 'pandas',
                        # 'scalpl', # remove it
                        ]  # And any other dependencies foo needs
      )
