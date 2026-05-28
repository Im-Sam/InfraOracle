import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

# =========================
# CONFIG
# =========================

OUTPUT_PATH = "data/demo/demo_vm_metrics.csv"

NUM_VMS = 50
DAYS = 7
INTERVAL_MINUTES = 10

np.random.seed(42)
random.seed(42)

# =========================
# VM ROLES (REAL INFRA PATTERNS)
# =========================

VM_ROLES = [
    "web-server",
    "api-gateway",
    "database-primary",
    "database-replica",
    "cache-node",
    "batch-worker",
    "logging-node",
    "monitoring-agent"
]

# =========================
# ROLE BEHAVIOR PROFILES
# =========================

ROLE_PROFILES = {
    "web-server": {
        "cpu_base": 35,
        "mem_base": 45,
        "cpu_noise": 10,
        "mem_noise": 8,
        "net_scale": 80000,
        "disk_scale": 1200
    },
    "api-gateway": {
        "cpu_base": 50,
        "mem_base": 55,
        "cpu_noise": 15,
        "mem_noise": 10,
        "net_scale": 150000,
        "disk_scale": 900
    },
    "database-primary": {
        "cpu_base": 60,
        "mem_base": 75,
        "cpu_noise": 8,
        "mem_noise": 5,
        "net_scale": 40000,
        "disk_scale": 5000
    },
    "database-replica": {
        "cpu_base": 45,
        "mem_base": 65,
        "cpu_noise": 6,
        "mem_noise": 5,
        "net_scale": 35000,
        "disk_scale": 4500
    },
    "cache-node": {
        "cpu_base": 20,
        "mem_base": 80,
        "cpu_noise": 5,
        "mem_noise": 10,
        "net_scale": 60000,
        "disk_scale": 300
    },
    "batch-worker": {
        "cpu_base": 70,
        "mem_base": 60,
        "cpu_noise": 20,
        "mem_noise": 15,
        "net_scale": 30000,
        "disk_scale": 2000
    },
    "logging-node": {
        "cpu_base": 25,
        "mem_base": 50,
        "cpu_noise": 6,
        "mem_noise": 5,
        "net_scale": 120000,
        "disk_scale": 6000
    },
    "monitoring-agent": {
        "cpu_base": 10,
        "mem_base": 30,
        "cpu_noise": 2,
        "mem_noise": 2,
        "net_scale": 20000,
        "disk_scale": 200
    }
}

# =========================
# GENERATE VM INVENTORY
# =========================

vms = []

for i in range(NUM_VMS):
    role = random.choice(VM_ROLES)

    vms.append({
        "vm_id": f"vm-{role}-{i:03d}",
        "role": role,
        "failure_mode": "healthy" if random.random() > 0.15 else "degrading"
    })

# =========================
# TIME SERIES GENERATION
# =========================

start_time = datetime.now() - timedelta(days=DAYS)
total_steps = DAYS * 24 * 60 // INTERVAL_MINUTES

rows = []

for vm in vms:

    profile = ROLE_PROFILES[vm["role"]]

    for t in range(total_steps):

        timestamp = start_time + timedelta(minutes=t * INTERVAL_MINUTES)

        # =========================
        # BASE LOAD
        # =========================

        cpu = np.random.normal(profile["cpu_base"], profile["cpu_noise"])
        mem = np.random.normal(profile["mem_base"], profile["mem_noise"])

        disk = np.random.normal(profile["disk_scale"], profile["disk_scale"] * 0.1)
        net = np.random.normal(profile["net_scale"], profile["net_scale"] * 0.15)

        # =========================
        # ROLE-SPECIFIC BEHAVIOR
        # =========================

        if vm["role"] == "api-gateway":
            if np.random.rand() < 0.05:
                cpu += np.random.randint(20, 40)

        if vm["role"] == "batch-worker":
            if (t // 100) % 2 == 0:
                cpu += 15

        if vm["role"] == "cache-node":
            mem += cpu * 0.5

        if vm["role"] == "database-primary":
            disk += cpu * 50

        # =========================
        # FAILURE SIMULATION
        # =========================

        if vm["failure_mode"] == "degrading":

            if t > total_steps * 0.7:

                drift = (t - total_steps * 0.7)

                cpu += drift * 0.02
                mem += drift * 0.015
                disk += drift * 2

            if t > total_steps * 0.9 and np.random.rand() < 0.03:
                cpu += 30
                mem += 25
                disk += 5000

        # =========================
        # CLAMP VALUES
        # =========================

        cpu = max(0, min(100, cpu))
        mem = max(0, min(100, mem))

        rows.append([
            timestamp,
            vm["vm_id"],
            vm["role"],
            round(cpu, 2),
            round(mem, 2),
            round(disk, 2),
            round(net, 2)
        ])

# =========================
# SAVE
# =========================

df = pd.DataFrame(rows, columns=[
    "timestamp",
    "vm_id",
    "role",
    "cpu_pct",
    "mem_pct",
    "disk_io",
    "net_bytes"
])

os.makedirs("data/demo", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("Generated 50 realistic VMs with infrastructure roles.")