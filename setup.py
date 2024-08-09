from setuptools import setup, find_packages


setup(
    name='random_forest_package',
    version='0.1.4',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'numpy',
        'pandas',
        'flake8',
        'lint',
        'matplotlib',
        'seaborn'
    ],
    author='Karim Mirzaguliyev',
    author_email='karimmirzaguliyev@gmail.com',
    description='A package to facilitate random forest modeling',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/k4rimDev/random_forest_package',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
