# %% [markdown]
# ## Inhabilitar uso de threads en numpy
#
# Inhabilitar cuando se use procesos físicos para evitar competencia de recursos
#
# Ejecutar antes de import numpy
#
# **Si se ejecuto con threads y se quiere ejecutar con cpu física primero se debe reiniciar el kernel y ejecutar la celda, igualmente si se ejecuto la celda y quiere activar de nuevo los threads para numpy**

from __future__ import annotations

import os
import platform
import time
from functools import wraps

import cpuinfo
import psutil
import torch

# %%
# import os
# # ── Configuración de threads ──────────────────────────────
# os.environ["OMP_NUM_THREADS"]     = "1"
# os.environ["MKL_NUM_THREADS"]     = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# ## Librerías - utilidades
# %%


def print_system_info():
    print("=== SYSTEM INFO ===")
    print("OS:", platform.system(), platform.release())
    print("Machine:", platform.machine())

    info = cpuinfo.get_cpu_info()
    print("CPU:", info["brand_raw"])
    print("Architecture:", info["arch"])
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Logical threads:", psutil.cpu_count(logical=True))

    ram = psutil.virtual_memory()
    print("RAM (GB):", round(ram.total / (1024**3), 2))
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"cpu_count: {os.cpu_count()}")

    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM Total (GB): {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}"
        )
        print(f"VRAM Allocated (GB): {torch.cuda.memory_allocated(0) / 1024**3:.2f}")
        print(f"VRAM Reserved (GB): {torch.cuda.memory_reserved(0) / 1024**3:.2f}")

    print("_" * 40)


def time_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if elapsed < 1:
            formatted = f"{elapsed * 1000:.2f} ms"
        elif elapsed < 60:
            formatted = f"{elapsed:.2f} s"
        elif elapsed < 3600:
            mins, secs = divmod(elapsed, 60)
            formatted = f"{int(mins)}m {secs:.2f}s"
        else:
            hours, remainder = divmod(elapsed, 3600)
            mins, secs = divmod(remainder, 60)
            formatted = f"{int(hours)}h {int(mins)}m {secs:.2f}s"

        print(f"{func.__name__} took {formatted}")
        return result

    return wrapper


__all__ = ["print_system_info", "time_wrapper"]
