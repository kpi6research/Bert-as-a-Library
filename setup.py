from distutils.core import setup
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
  name = 'BertLib',         # How you named your package folder (MyLib)
  packages = ['BertLib'],   # Chose the same as "name"
  version = '0.2',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'BaaL is a Tensorflow library for quick and easy training and finetuning of models based on Bert',   # Give a short description about your library
  author = 'Andrea Salvoni',                   # Type in your name
  author_email = 'andrea.salvoni93@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/kpi6research/Bert-as-a-Library',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/kpi6research/Bert-as-a-Library/archive/v_02.tar.gz',    # I explain this later on
  keywords = ['Bert', 'fientuning', 'nlp'],   # Keywords that define your package best
  install_requires=install_deps,
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)