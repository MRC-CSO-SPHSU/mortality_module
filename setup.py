"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="mortality_module",
    version="0.0.1",
    description="Population dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    # author="A. Random Developer",
    # author_email="author@example.com",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="simulation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=["numpy", "pandas"],
    extras_require={
        "dev": ["check-manifest"],
        "test": ["coverage"],
    },
    # package_data={"sample": ["package_data.dat"],},
    # data_files=[("my_data", ["data/data_file"])],
)
