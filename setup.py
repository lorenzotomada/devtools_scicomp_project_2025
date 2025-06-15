import os
import sys
import shutil
import glob
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        build_temp = os.path.abspath(self.build_temp)
        build_lib = os.path.abspath(self.build_lib)

        # Run CMake build
        os.makedirs(build_temp, exist_ok=True)
        self.spawn(["cmake", ext.sourcedir, "-B", build_temp])
        self.spawn(["cmake", "--build", build_temp, "--target", "cxx_utils"])

        # Dynamically find the compiled shared library
        matches = glob.glob(
            os.path.join(ext.sourcedir, "src", "pyclassify", "cxx_utils*.so")
        )
        if not matches:
            raise RuntimeError(
                "Could not find compiled cxx_utils shared library in expected location."
            )

        src_lib = os.path.abspath(matches[0])
        dst_lib = os.path.join(build_lib, "pyclassify", os.path.basename(src_lib))

        os.makedirs(os.path.dirname(dst_lib), exist_ok=True)
        shutil.copy(src_lib, dst_lib)


setup(
    name="pyclassify",
    version="0.0.1",
    author="Gaspare Li Causi, Lorenzo Tomada",
    author_email="ltomada@sissa.it, glicausi@sissa.it",
    description="Final project",
    long_description="Eigenvalue computation",
    ext_modules=[CMakeExtension("pyclassify.cxx_utils")],
    packages=find_packages(where="src/"),
    package_dir={"": "src/"},
    package_data={"pyclassify": ["cxx_utils*.so"]},
    include_package_data=True,
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
