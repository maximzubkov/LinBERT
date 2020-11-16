#!/usr/bin/env python
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Setup fast transformers"""

from functools import lru_cache
from itertools import dropwhile
from os import path
from subprocess import DEVNULL, call

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


@lru_cache(None)
def cuda_toolkit_available():
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def collect_metadata():
    meta = {}
    with open(path.join("models", "modules", "fast_transformers", "__init__.py")) as f:
        lines = iter(f)
        meta["description"] = collect_docstring(lines)
        for line in lines:
            if line.startswith("__"):
                key, value = map(lambda x: x.strip(), line.split("="))
                meta[key[2:-2]] = value[1:-1]

    return meta


def get_extensions():
    extensions = [
        CppExtension(
            "models.modules.fast_transformers.causal_product.causal_product_cpu",
            sources=[
                "models/modules/fast_transformers/causal_product/causal_product_cpu.cpp"
            ],
            extra_compile_args=["-Xpreprocessor", "-fopenmp", "-ffast-math"],
        )
    ]
    if cuda_toolkit_available():
        from torch.utils.cpp_extension import CUDAExtension

        extensions += [
            CUDAExtension(
                "models.modules.fast_transformers.causal_product.causal_product_cuda",
                sources=[
                    "models/modules/fast_transformers/causal_product/causal_product_cuda.cu"
                ],
                extra_compile_args=["-arch=compute_50"],
            )
        ]
    return extensions


def setup_package():
    meta = collect_metadata()
    print(meta)
    setup(
        name="pytorch-fast-transformers",
        version="0.0.1",
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
        ],
        packages=find_packages(exclude=["docs", "tests", "scripts", "examples"]),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension},
    )


if __name__ == "__main__":
    setup_package()
