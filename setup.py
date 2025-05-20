import os
import sys
import shutil
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

        python_version = sys.version_info
        so_filename = (
            f"QR_cpp.cpython-{python_version[0]}{python_version[1]}-x86_64-linux-gnu.so"
        )

        src_lib = os.path.join(build_temp, f"../../src/pyclassify/{so_filename}")
        dst_lib = os.path.join(build_lib, f"pyclassify/{so_filename}")

        if os.path.exists(src_lib):
            os.makedirs(os.path.dirname(dst_lib), exist_ok=True)
            shutil.copy(src_lib, dst_lib)
        else:
            raise RuntimeError(
                f"Could not find compiled QR_cpp shared library at {src_lib}!"
            )


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
    package_data={"pyclassify": ["QR_cpp.cpython-312-x86_64-linux-gnu.so"]},
    include_package_data=True,
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
