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
    requires=['numpy', 'scipy', 'george', 'pandas', 'matplotlib']
)
