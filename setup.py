import os
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
  Pybind11Extension(
    "ood_prompts.ext.count_utils",
    ["cpp/count_utils.cpp"],
    extra_compile_args=["-std=c++11"],
  ),
]


class CustomBuildExt(build_ext):
  def run(self):
    ext_dir = os.path.join("ood_prompts", "ext")
    os.makedirs(ext_dir, exist_ok=True)

    init_path = os.path.join(ext_dir, "__init__.py")
    if not os.path.exists(init_path):
      open(init_path, "w").close()

    super().run()


setup(
  name="ood_prompts",
  packages=find_packages(),
  ext_modules=ext_modules,
  cmdclass={"build_ext": CustomBuildExt},
)
