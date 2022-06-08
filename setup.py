import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qsc",
    version="0.1.1",
    author="Matt Landreman",
    author_email="matt.landreman@gmail.com",
    description="Quasisymmetric Stellarator Construction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/landreman/pyQSC",
    #install_requires=['numpy', 'scipy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
#    packages=["qsc"],
