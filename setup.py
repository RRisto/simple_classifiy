from setuptools import setup

def readme():
    with open('readme.md') as f:
        return f.read()

setup(name='TextClass',
      version='0.1',
      description='Classes and methods for cleaning, cleaning and classifyng texts',
      long_description=readme(),
      classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.5',
        'Topic :: Text Processing :: General',
      ],
      keywords='nlp classification',
      url='https://github.com/RRisto/simple_classifiy',
      author='Risto Hinno',
      author_email='ristohinno@gmail.com',
      license='MIT',
      packages=['TextClass'],
      install_requires=['nltk','sklearn','imblearn','gensim','matplotlib','pyldavis','pandas','pytest','openpyxl',
                        'mock', 'fasttext==0.8.3'],
      test_suite='pytest-runner',
      tests_require=['pytest'],
      include_package_data=True,
      zip_safe=False)