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
        self.spawn(["cmake", "--build", build_temp, "--target", "QR_cpp"])

        # Dynamically find the compiled shared library
        matches = glob.glob(os.path.join(ext.sourcedir, "src", "pyclassify", "QR_cpp*.so"))
        if not matches:
            raise RuntimeError(
                "Could not find compiled QR_cpp shared library in expected location."
            )

        src_lib = os.path.abspath(matches[0])
        dst_lib = os.path.join(build_lib, "pyclassify", os.path.basename(src_lib))

        os.makedirs(os.path.dirname(dst_lib), exist_ok=True)
        shutil.copy(src_lib, dst_lib)


setup(
    name="pyclassify",
    version="0.0.1",
    author="Lorenzo Tomada, Gaspare Li Causi",
    author_email="ltomada@sissa.it, glicausi@sissa.it",
    description="Final project",
    long_description="Eigenvalue computation",
    ext_modules=[CMakeExtension("pyclassify.QR_cpp")],
    packages=find_packages(where="src/"),
    package_dir={"": "src/"},
    package_data={"pyclassify": ["QR_cpp*.so"]},
    include_package_data=True,
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
