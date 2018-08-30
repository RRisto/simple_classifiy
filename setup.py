from setuptools import setup


def readme():
    with open('README.txt') as f:
        return f.read()


setup(name='simple_classify',
      version='0.1',
      description='Classes and methods for cleaning, cleaning and classifiyng texts',
      long_description=readme(),
      classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.4',
        'Topic :: Text Processing :: General',
      ],
      keywords='nlp cleaning text',
      url='',
      author='Feelingstream',
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=['simple_classify'],
      install_requires=[
          'nltk','langdetect','estnltk', 'pytest'
      ],
      test_suite='pytest-runner',
      tests_require=['pytest'],
      include_package_data=True,
      zip_safe=False)