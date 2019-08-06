from setuptools import setup, find_packages
from pkg_resources import DistributionNotFound, get_distribution


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


install_deps = ['numpy']

if get_dist('tensorflow') is None and get_dist('tensorflow_gpu') is None:
    install_deps.append('tensorflow')

setup(
  name = 'BertLibrary',   
  packages = find_packages(),
  version = '0.0.1',     
  license='MIT',        
  description = 'BaaL is a Tensorflow library for quick and easy training and finetuning of models based on Bert', 
  author = 'KPI6 Research',                   
  author_email = 'info@kpi6.com',    
  url = 'https://github.com/kpi6research/Bert-as-a-Library',   
  download_url = 'https://github.com/kpi6research/Bert-as-a-Library/archive/0.0.1.tar.gz',   
  keywords = ['Bert', 'fientuning', 'nlp'],  
  install_requires=install_deps,
  classifiers=[
    'Development Status :: 4 - Beta',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
