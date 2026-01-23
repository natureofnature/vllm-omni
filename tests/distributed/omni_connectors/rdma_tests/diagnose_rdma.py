# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
RDMA Diagnostic Script

This script checks the RDMA environment and reports:
1. Available RDMA devices and their status
2. Mooncake TransferEngine availability
3. Network configuration
4. Recommended settings for testing

Usage:
    python diagnose_rdma.py
"""

import json
import os
import subprocess
import sys


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_status(name: str, status: bool, detail: str = ""):
    """Print a status line."""
    icon = "✓" if status else "✗"
    color_start = "\033[92m" if status else "\033[91m"
    color_end = "\033[0m"
    print(f"  {color_start}{icon}{color_end} {name}: {detail}")


def check_mooncake():
    """Check if Mooncake TransferEngine is available."""
    print_header("Mooncake TransferEngine Check")

    try:
        from mooncake.engine import TransferEngine

        print_status("Mooncake import", True, "mooncake.engine.TransferEngine available")

        # Try to initialize
        engine = TransferEngine()
        ret = engine.initialize("127.0.0.1", "P2PHANDSHAKE", "rdma", "")
        if ret == 0:
            print_status("TransferEngine init", True, "Initialized successfully")

            # Get topology
            topo_str = engine.get_local_topology()
            if topo_str:
                topo = json.loads(topo_str)
                devices = list(topo.keys())
                print_status("Topology discovery", True, f"Found {len(devices)} devices: {devices}")
                return True, devices
            else:
                print_status("Topology discovery", False, "No topology returned")
        else:
            print_status("TransferEngine init", False, f"Failed with code {ret}")

    except ImportError as e:
        print_status("Mooncake import", False, f"Not installed: {e}")
    except Exception as e:
        print_status("Mooncake init", False, f"Error: {e}")

    return False, []


def check_ibstat():
    """Check InfiniBand devices using ibstat."""
    print_header("InfiniBand Devices (ibstat)")

    try:
        result = subprocess.run(["ibstat"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print_status("ibstat", False, result.stderr)
    except FileNotFoundError:
        print_status("ibstat", False, "Command not found. Install infiniband-diags.")
    except Exception as e:
        print_status("ibstat", False, str(e))

    return False


def check_ibdev2netdev():
    """Check RDMA device to network device mapping."""
    print_header("RDMA to Network Device Mapping")

    try:
        result = subprocess.run(["ibdev2netdev"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                print(f"  {line}")
            return True
        else:
            print_status("ibdev2netdev", False, result.stderr)
    except FileNotFoundError:
        print_status("ibdev2netdev", False, "Command not found")
    except Exception as e:
        print_status("ibdev2netdev", False, str(e))

    return False


def check_network_interfaces():
    """Check network interfaces for RDMA-capable IPs."""
    print_header("Network Interfaces")

    try:
        result = subprocess.run(["ip", "addr", "show"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Parse and show relevant interfaces
            current_iface = None
            for line in result.stdout.split("\n"):
                if not line.startswith(" "):
                    # Interface line
                    parts = line.split(":")
                    if len(parts) >= 2:
                        current_iface = parts[1].strip()
                elif "inet " in line and current_iface:
                    # IP address line
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        ip = parts[1]
                        # Skip loopback and docker
                        if not ip.startswith("127.") and "docker" not in current_iface:
                            print(f"  {current_iface}: {ip}")
            return True
    except Exception as e:
        print_status("ip addr", False, str(e))

    return False


def check_environment_variables():
    """Check RDMA-related environment variables."""
    print_header("Environment Variables")

    env_vars = [
        ("RDMA_DEVICE_NAME", "Specifies RDMA device to use"),
        ("RDMA_TEST_HOST", "Specifies test host IP"),
        ("MC_TE_METRIC", "Enable Mooncake metrics"),
        ("MC_IB_PCI_RELAXED_ORDERING", "Enable PCIe relaxed ordering"),
    ]

    for var, desc in env_vars:
        value = os.environ.get(var)
        if value:
            print_status(var, True, f"'{value}' ({desc})")
        else:
            print(f"  - {var}: not set ({desc})")


def check_docker_permissions():
    """Check if running in Docker and permissions."""
    print_header("Container/Permission Check")

    # Check if in Docker
    in_docker = os.path.exists("/.dockerenv")
    print_status("Running in Docker", in_docker, "Yes" if in_docker else "No (bare metal)")

    # Check /dev/infiniband
    ib_dev = os.path.exists("/dev/infiniband")
    print_status("/dev/infiniband", ib_dev, "Accessible" if ib_dev else "Not found")

    # Check /sys/class/infiniband
    ib_sys = os.path.exists("/sys/class/infiniband")
    print_status("/sys/class/infiniband", ib_sys, "Accessible" if ib_sys else "Not found")

    if ib_sys:
        try:
            devices = os.listdir("/sys/class/infiniband")
            print(f"    Devices: {devices}")
        except Exception as e:
            print(f"    Error listing devices: {e}")


def print_recommendations(mooncake_ok: bool, devices: list):
    """Print recommendations based on checks."""
    print_header("Recommendations")

    if not mooncake_ok:
        print("  1. Install Mooncake: pip install mooncake")
        print("  2. Ensure RDMA drivers are installed")
        return

    if len(devices) == 0:
        print("  1. Check InfiniBand driver installation")
        print("  2. Verify RDMA device permissions")
        return

    if len(devices) == 1:
        print(f"  Single device detected: {devices[0]}")
        print("  For single-node testing, this is ideal.")
        print("  Run tests: pytest test_rdma_correctness.py -v -s")
    else:
        print(f"  Multiple devices detected: {devices}")
        print("  For single-node testing, specify one device:")
        print(f"    export RDMA_DEVICE_NAME='{devices[0]}'")
        print("  For multi-node testing, ensure same device on both nodes.")

    print()
    print("  Quick test command:")
    if len(devices) > 1:
        print(f"    RDMA_DEVICE_NAME={devices[0]} pytest test_rdma_correctness.py -v -s")
    else:
        print("    pytest test_rdma_correctness.py -v -s")


def main():
    print("\n" + "=" * 60)
    print(" RDMA Environment Diagnostic Tool")
    print("=" * 60)

    # Run all checks
    check_environment_variables()
    check_docker_permissions()
    check_network_interfaces()
    check_ibdev2netdev()

    # Mooncake check (most important)
    mooncake_ok, devices = check_mooncake()

    # Recommendations
    print_recommendations(mooncake_ok, devices)

    print()
    return 0 if mooncake_ok else 1


if __name__ == "__main__":
    sys.exit(main())
