from pathlib import Path
import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent
PKG_NAME = "wginfer"
LIB_DIR = PKG_ROOT / PKG_NAME / "libwginfer"
BINDINGS_DIR = PKG_ROOT / "bindings"
BINDING_SOURCES = sorted(str(path) for path in BINDINGS_DIR.rglob("*.cpp"))

# Make `python python/setup.py build_ext --inplace` and
# `cd python && python setup.py build_ext --inplace` behave the same.
os.chdir(PKG_ROOT)


def linux_rpath():
    if sys.platform.startswith("linux"):
        return ["$ORIGIN/libwginfer"]
    return []


class WginferBuildExt(build_ext):
    def run(self):
        shared_lib = LIB_DIR / "libwginfer.so"
        if sys.platform.startswith("linux") and not shared_lib.exists():
            raise RuntimeError(
                "Missing python/wginfer/libwginfer/libwginfer.so. "
                "Build the core library first with CMake, for example "
                "`cmake -S . -B build && cmake --build build -j`, "
                "then ensure libwginfer.so is copied into python/wginfer/libwginfer/."
            )
        super().run()


ext_modules = [
    Pybind11Extension(
        "wginfer._wginfer",
        BINDING_SOURCES,
        include_dirs=[
            str(REPO_ROOT / "include"),
            str(REPO_ROOT / "src"),
            str(BINDINGS_DIR),
        ],
        libraries=["wginfer"],
        library_dirs=[str(LIB_DIR)],
        runtime_library_dirs=linux_rpath(),
        cxx_std=17,
    )
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": WginferBuildExt},
)
