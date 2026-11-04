"""Hardware and Environment Inspection Script for SLM Python Learning Assistant.

This script queries the host operating system, processor architecture,
physical memory, storage availability across drives, and display/GPU adapters.
It is designed to run in standard Python 3.10+ environments without requiring
external dependencies (uses standard library and Windows kernel32/winreg APIs).
"""

import ctypes
import os
import platform
import shutil
import subprocess
import sys
import winreg


def audit_system():
    print("=" * 60)
    print("SLM SYSTEM & HARDWARE AUDIT REPORT")
    print("=" * 60)

    # 1. Python Environment
    print("\n[1] Python & OS Environment")
    print(f"  Python Executable : {sys.executable}")
    print(f"  Python Version    : {sys.version.split()[0]}")
    print(f"  OS Platform       : {platform.platform()}")
    print(f"  Architecture      : {platform.machine()}")

    # 2. CPU Details
    print("\n[2] CPU Architecture")
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        proc_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        identifier, _ = winreg.QueryValueEx(key, "Identifier")
        mhz, _ = winreg.QueryValueEx(key, "~MHz")
        winreg.CloseKey(key)
        print(f"  Processor Name    : {proc_name.strip()}")
        print(f"  Identifier        : {identifier}")
        print(f"  Base Speed        : {mhz} MHz")
    except Exception as exc:
        print(f"  Could not read CPU registry: {exc}")
    print(f"  Logical Cores     : {os.cpu_count()}")

    # 3. GPU / Display Adapters
    print("\n[3] Graphics & GPU Adapters")
    gpu_found = False
    try:
        video_class = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, video_class)
        i = 0
        while True:
            try:
                subkey_name = winreg.EnumKey(key, i)
                i += 1
                if subkey_name.isdigit():
                    sub_key = winreg.OpenKey(key, subkey_name)
                    try:
                        desc, _ = winreg.QueryValueEx(sub_key, "DriverDesc")
                        provider, _ = winreg.QueryValueEx(sub_key, "ProviderName")
                        ver, _ = winreg.QueryValueEx(sub_key, "DriverVersion")
                        print(f"  Adapter [{subkey_name}]     : {desc} (Provider: {provider}, Ver: {ver})")
                        gpu_found = True
                    except Exception:
                        pass
                    winreg.CloseKey(sub_key)
            except OSError:
                break
        winreg.CloseKey(key)
    except Exception as exc:
        print(f"  Could not read video adapter registry: {exc}")

    # Check for NVIDIA CUDA specifically
    cuda_path = os.environ.get("CUDA_PATH", None)
    print(f"  CUDA_PATH Env Var : {cuda_path if cuda_path else 'Not detected'}")

    has_nv = False
    for p in [r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe", r"C:\Windows\System32\nvidia-smi.exe"]:
        if os.path.exists(p):
            has_nv = True
            print(f"  nvidia-smi found  : {p}")
    if not has_nv:
        print("  Dedicated NVIDIA  : Not detected (System uses Integrated Graphics)")

    # 4. Memory Details
    print("\n[4] Physical & Virtual Memory")
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_ram_gb = stat.ullTotalPhys / (1024**3)
        avail_ram_gb = stat.ullAvailPhys / (1024**3)
        print(f"  Total Physical RAM: {total_ram_gb:.2f} GB")
        print(f"  Available RAM     : {avail_ram_gb:.2f} GB")
        print(f"  Memory Load       : {stat.dwMemoryLoad}%")
        print(f"  Total Page File   : {stat.ullTotalPageFile / (1024**3):.2f} GB")
        print(f"  Available Pagefile: {stat.ullAvailPageFile / (1024**3):.2f} GB")
    except Exception as exc:
        print(f"  Memory detection failed: {exc}")

    # 5. Storage Drives
    print("\n[5] Storage Drives")
    for drive in ["C:\\", "D:\\"]:
        if os.path.exists(drive):
            total, used, free = shutil.disk_usage(drive)
            print(f"  Drive {drive}          : Total: {total/(1024**3):.2f} GB | Used: {used/(1024**3):.2f} GB | Free: {free/(1024**3):.2f} GB")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    audit_system()
