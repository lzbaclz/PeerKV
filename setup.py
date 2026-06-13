"""Build hook for the optional UMA-LLM native CUDA extension.

Project metadata lives in ``pyproject.toml``; this file exists only to build
``umallm._uma_native`` (the Route-B managed KV allocator + residency control)
when a CUDA toolchain is present. On a CPU/Mac host the extension is skipped
and ``pip install .`` still works -- the Python layer falls back to a
simulation (see ``umallm/uma_alloc.py``).

Force on/off with the ``UMA_BUILD_CUDA`` env var (``1``/``0``); unset = auto.

Building the CUDA extension needs PyTorch *at build time*, which pip's default
build isolation hides -- so install torch first and pass --no-build-isolation:

    pip install "torch>=2.4"
    UMA_BUILD_CUDA=1 pip install -e . --no-build-isolation              # A100+GH200
    UMA_BUILD_CUDA=1 UMA_CUDA_ARCH=80 pip install -e . --no-build-isolation  # A100 only
"""
import os
import shutil

from setuptools import setup


def _want_cuda() -> bool:
    flag = os.environ.get("UMA_BUILD_CUDA")
    if flag is not None:
        return flag.strip() not in ("0", "", "false", "False")
    # Auto: build only if torch is importable with CUDA and nvcc is on PATH.
    if shutil.which("nvcc") is None:
        return False
    try:
        import torch  # noqa: F401

        return bool(torch.cuda.is_available()) or os.path.isdir(
            os.environ.get("CUDA_HOME", "/usr/local/cuda")
        )
    except Exception:
        return False


def _ext_modules():
    if not _want_cuda():
        print("UMA-LLM: CUDA toolchain not detected -- skipping _uma_native "
              "(Route B will use the CPU simulation).")
        return [], {}
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError as e:  # torch absent at build time (pip build isolation?)
        raise SystemExit(
            "UMA-LLM: building _uma_native needs PyTorch at build time, but "
            "`import torch` failed. Install torch first and disable build "
            "isolation:\n"
            "    pip install 'torch>=2.4'\n"
            "    UMA_BUILD_CUDA=1 pip install -e . --no-build-isolation\n"
            f"(original error: {e})")

    # Target arches: default A100 (sm_80) + GH200 (sm_90); override with
    # UMA_CUDA_ARCH="80", "90", or "80;90". PTX of the highest arch is kept so
    # the build also runs on newer GPUs.
    arches = [a.strip() for a in
              os.environ.get("UMA_CUDA_ARCH", "80;90").replace(",", ";").split(";")
              if a.strip()]
    if not arches:
        arches = ["80", "90"]
    nvcc = ["-O3"]
    for a in arches:
        nvcc.append(f"-gencode=arch=compute_{a},code=sm_{a}")
    ptx = max(arches, key=lambda a: int(a))
    nvcc.append(f"-gencode=arch=compute_{ptx},code=compute_{ptx}")
    print(f"UMA-LLM: building _uma_native for CUDA arch(es) {arches} (+PTX {ptx}).")

    ext = CUDAExtension(
        name="umallm._uma_native",
        sources=[
            "csrc/bindings.cpp",
            "csrc/uma_managed_alloc.cu",
            "csrc/uma_residency.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-fvisibility=default"],
            "nvcc": nvcc,
        },
    )
    return [ext], {"build_ext": BuildExtension}


ext_modules, cmdclass = _ext_modules()

setup(ext_modules=ext_modules, cmdclass=cmdclass)
