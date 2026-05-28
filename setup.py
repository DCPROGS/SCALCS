from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="scalcs",
    version="0.5.0",
    description="Q-matrix calculations of ion channel propeties",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/remislp/SCALCS",  # mature version will be moved to https://github.com/DCPROGS/SCALCS
    author="Remis Lape",
    author_email="remis.lp@gmail.com",

    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="scalcs, ion-channels, matrix",
    packages=find_packages(where="src"),  # Required
    python_requires=">=3.10, <4",
    install_requires=["numpy", "matplotlib", "pandas", "scipy", "PyQt5", "pyyaml"],
)
