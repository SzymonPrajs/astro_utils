from distutils.core import setup

setup(
    name='astrotools',
    version='0.1',
    packages=['astrotools'],
    url='github.com/szymonprajs/astrotools',
    license='MIT',
    author='Szymon Prajs',
    author_email='S.Prajs@soton.ac.uk',
    description='Set of astronomy tools and wrappers developed over the durations my PhD',
    requires=['numpy', 'scipy', 'george', 'pandas', 'matplotlib', 'astropy'],
    keywords=['astronomy', 'astrophysics', 'cosmology', 'space', 'science',
              'units', 'table', 'modeling', 'models', 'fitting', 'ascii'],
    classifiers=['Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT',
                 'Operating System :: OS Independent',
                 'Programming Language :: C',
                 'Programming Language :: Cython',
                 'Programming Language :: Python :: 3',
                 'Topic :: Scientific/Engineering :: Astronomy']
    # python_requires='>=3.4'
)