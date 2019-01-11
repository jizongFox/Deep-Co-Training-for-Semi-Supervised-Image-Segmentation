import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="generalframework",
    version='0.0.1',
    author="Jizong Peng",
    author_email="jizong.peng.1@etsmtl.net",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages()
)
