#!/usr/bin/env python3
"""
KoboldCPP Benchmark Optimizer MCP Server

This server benchmarks LLMs. It requires the path for the gguf model weights.

Usage:
1. analyze model and system - Connect to remote system
2. run benchmark - Run complete optimization for a model

All operations happen on the remote system via SSH - no local model files needed.
"""

import asyncio
import subprocess
import json
import re
import os
import hashlib
import struct
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import paramiko
from fastmcp import FastMCP
from tinydb import TinyDB, Query as TinyQuery

# You can set SSH host and credentials in env
# HOST = os.getenv("SSH_HOST")
# USERNAME = os.getenv("SSH_USERNAME")
# PASSWORD = os.getenv("SSH_PASSWORD")
# GGUF_DUMP = Path(os.getenv("GGUF_DUMP_LOCATION")

# Or put SHH host and credentials here
HOST = ""
USERNAME = ""
PASSWORD = ""

# Point this to Koboldcpp python file
GGUF_DUMP = "~/koboldcpp/koboldcpp.py"

#The model to test
MODEL_PATH = ""

class BenchmarkOptimizer:
    """Main optimizer class"""

    def __init__(self, db_path: str = "benchmark_results.json"):
        self.db = TinyDB(db_path)
        self.models_table = self.db.table("models")
        self.experiments_table = self.db.table("experiments")
        self.system_table = self.db.table("system_info")

        # SSH connection details
        self.ssh_client = None
        self.ssh_host = None
        self.ssh_user = None
        self.ssh_password = None

    def get_model_hash(self, model_path: str) -> str:
        """Create consistent model identifier"""
        return hashlib.md5(model_path.encode()).hexdigest()[:16]

    async def detect_system_capabilities(self) -> Dict[str, Any]:
        """Discover remote system VRAM and capabilities via SSH"""
        if not self.ssh_client:
            #self.setup_ssh_connection()
            raise RuntimeError("SSH connection not established")

        try:
            # Get CUDA version and availability
            stdin, stdout, stderr = self.ssh_client.exec_command(
                "nvidia-smi -q -d=compute"
            )
            output = stdout.read().decode("utf-8")
            error_output = stderr.read().decode("utf-8")

            if stdout.channel.recv_exit_status() != 0:
                raise RuntimeError(f"nvidia-smi failed: {error_output}")

            system_info = {
                "timestamp": datetime.utcnow().isoformat(),
                "cuda_available": False,
                "cuda_version": None,
                "total_vram_mb": 0,
                "gpu_count": 0,
                "gpu_names": [],
            }

            for line in output.splitlines():
                if line.strip().startswith("CUDA"):
                    system_info["cuda_version"] = line.split()[3]
                    system_info["cuda_available"] = True

            # Get VRAM information
            stdin, stdout, stderr = self.ssh_client.exec_command(
                "nvidia-smi --query-gpu=memory.total --format=csv,noheader"
            )
            output = stdout.read().decode("utf-8")

            if stdout.channel.recv_exit_status() != 0:
                raise RuntimeError("Failed to get VRAM info")

            vram_mb = 0
            gpu_count = 0
            for line in output.splitlines():
                if line.strip():
                    vram_mb += int(line.strip().split()[0])
                    gpu_count += 1

            system_info["total_vram_mb"] = vram_mb
            system_info["gpu_count"] = gpu_count

            # Get GPU names
            stdin, stdout, stderr = self.ssh_client.exec_command(
                "nvidia-smi --query-gpu=name --format=csv,noheader"
            )
            output = stdout.read().decode("utf-8")

            system_info["gpu_names"] = [
                line.strip() for line in output.splitlines() if line.strip()
            ]

            # Store in database
            self.system_table.truncate()
            self.system_table.insert(system_info)

            return system_info

        except Exception as e:
            raise RuntimeError(f"Failed to detect remote NVIDIA GPUs: {e}")

    async def analyze_model(self, model_path: str) -> Dict[str, Any]:
        """Extract model metadata using GGUF reader on remote system via SSH"""
        result = {
            "path": str(model_path),
            "total_layers": 0,
            "model_name": "unknown",
            "context_length": 0,
            "architecture": None,
        }
        gguf_dump_args = "--analyze"
        if not self.ssh_client:
            #self.setup_ssh_connection()
            raise RuntimeError("SSH connection not established")

        # Execute script on remote system
        stdin, stdout, stderr = self.ssh_client.exec_command(
            f"python3 {GGUF_DUMP} {gguf_dump_args} '{model_path}'"
        )
        output = stdout.read().decode("utf-8")
        error_output = stderr.read().decode("utf-8")
        exit_status = stdout.channel.recv_exit_status()

        if exit_status != 0:
            raise RuntimeError(f"Remote GGUF analysis failed: {error_output}")
        
        for line in output.splitlines():
            architecture = result.get("architecture")
            if architecture is not None:
                if architecture + ".block_count" in line:
                    result["total_layers"] = int(line.split()[-1])
                elif "general.name" in line:
                    result["model_name"] = ' '.join(line.split()[3:])
                elif architecture + ".context_length" in line:
                    result["context_length"] = int(line.split()[-1])
            else:
                if "general.architecture" in line:
                    result["architecture"] = str(line.split()[-1])
                
        # Parse JSON result
        # remote_result = json.loads(output)
        remote_result = result
        if "error" in remote_result:
            raise RuntimeError(f"GGUF analysis error: {remote_result['error']}")

        # Add local fields
        model_info = {
            **remote_result,
            "model_hash": self.get_model_hash(model_path),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Store in database
        existing = self.models_table.get(
            TinyQuery().model_hash == model_info["model_hash"]
        )
        if existing:
            self.models_table.update(
                model_info, TinyQuery().model_hash == model_info["model_hash"]
            )
        else:
            self.models_table.insert(model_info)

        return model_info

    async def setup_ssh_connection(self):
        """Establish SSH connection to remote system"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            self.ssh_client.connect(HOST, username=USERNAME, password=PASSWORD)

            self.ssh_host = HOST
            self.ssh_user = USERNAME
            self.ssh_password = PASSWORD

            return {"status": "connected", "host": HOST, "user": USERNAME}

        except Exception as e:
            raise RuntimeError(f"SSH connection failed: {e}")

    async def run_benchmark_via_ssh(
        self,
        model_path: str,
        gpulayers: int,
        moecpu: int,
        contextsize: int,
        blasbatchsize: int = 512,
        other_args: List[str] = None,
    ) -> Dict[str, Any]:
        """Execute benchmark command via SSH and parse results
        
        Args:
            model_path: Path to model file on remote system
            gpulayers: Number of GPU layers to use
            moecpu: Number of MoE CPU layers
            contextsize: Context size for benchmark
            blasbatchsize: BLAS batch size (-1,16,32,64,128,256,512,1024,2048)
            other_args: Additional command line arguments
        """
        if not self.ssh_client:
            #self.setup_ssh_connection()
            raise RuntimeError("SSH connection not established")

        other_args = other_args or [
            "--usecublas",
            "rowsplit",
            "--multiuser",
            "--flashattention",
        ]

        # Validate blasbatchsize
        valid_blas_sizes = {-1, 16, 32, 64, 128, 256, 512, 1024, 2048}
        if blasbatchsize not in valid_blas_sizes:
            raise ValueError(f"Invalid blasbatchsize {blasbatchsize}. Must be one of {valid_blas_sizes}")

        # Build command
        cmd_parts = [
            "python3",
            "~/koboldcpp/koboldcpp.py",
            "--gpulayers",
            str(gpulayers),
            "--moecpu",
            str(moecpu),
            "--contextsize",
            str(contextsize),
            "--blasbatchsize",
            str(blasbatchsize),
            "--model",
            model_path,
            "--benchmark",
        ] + other_args

        base_command = " ".join(cmd_parts)
        command = f"cd /mnt/Orlando/gguf && {base_command}"

        try:
            # Execute command
            stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=300)

            # Wait for completion
            exit_status = stdout.channel.recv_exit_status()
            output = stdout.read().decode("utf-8")
            error_output = stderr.read().decode("utf-8")

            # Parse benchmark results
            results = self._parse_benchmark_output(output)
            if exit_status == 1 or exit_status == 0:
                experiment = {
                    "id": len(self.experiments_table) + 1,
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_path": model_path,
                    "model_hash": self.get_model_hash(model_path),
                    "parameters": {
                        "gpulayers": gpulayers,
                        "moecpu": moecpu,
                        "contextsize": contextsize,
                        "blasbatchsize": blasbatchsize,
                        "other_args": other_args,
                    },
                    "command_used": command,
                    "exit_status": exit_status,
                    #"raw_output": "",
                    #"error_output": "",
                    "results": results,
                }
            
            else:
                experiment = {
                    "id": len(self.experiments_table) + 1,
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_path": model_path,
                    "model_hash": self.get_model_hash(model_path),
                    "parameters": {
                        "gpulayers": gpulayers,
                        "moecpu": moecpu,
                        "contextsize": contextsize,
                        "blasbatchsize": blasbatchsize,
                        "other_args": other_args,
                    },
                    "command_used": command,
                    "exit_status": exit_status,
                    #"raw_output": output,
                    #"error_output": error_output,
                    "results": {"status": "error"},
                } 
            # Store experiment
            self.experiments_table.insert(experiment)
    
            return experiment

        except Exception as e:
            error_experiment = {
                "id": len(self.experiments_table) + 1,
                "timestamp": datetime.utcnow().isoformat(),
                "model_path": model_path,
                "model_hash": self.get_model_hash(model_path),
                "parameters": {
                    "gpulayers": gpulayers,
                    "moecpu": moecpu,
                    "contextsize": contextsize,
                    "blasbatchsize": blasbatchsize,
                    "other_args": other_args,
                },
                "command_used": command,
                "error_message": str(e),
                "results": {"status": "error"},
            }
            self.experiments_table.insert(error_experiment)
            raise RuntimeError(f"Benchmark execution failed: {e}")

    def _parse_benchmark_output(self, output: str) -> Dict[str, Any]:
        """Parse benchmark output and extract performance metrics"""
        results = {"status": "error"}

        try:
            # Look for memory usage information
            memory_info = {}
            for line in output.splitlines():
                if "KV buffer size" in line:
                    if "CPU" in line:
                        memory_info["cpu_kv_buffer_mb"] = float(
                            re.search(r"(\d+\.\d+) MiB", line).group(1)
                        )
                    elif "CUDA0" in line:
                        memory_info["cuda0_kv_buffer_mb"] = float(
                            re.search(r"(\d+\.\d+) MiB", line).group(1)
                        )
                    elif "CUDA1" in line:
                        memory_info["cuda1_kv_buffer_mb"] = float(
                            re.search(r"(\d+\.\d+) MiB", line).group(1)
                        )
                elif "compute buffer size" in line:
                    if "CUDA0" in line:
                        memory_info["cuda0_compute_buffer_mb"] = float(
                            re.search(r"(\d+\.\d+) MiB", line).group(1)
                        )
                    elif "CUDA1" in line:
                        memory_info["cuda1_compute_buffer_mb"] = float(
                            re.search(r"(\d+\.\d+) MiB", line).group(1)
                        )

            # Look for benchmark results
            benchmark_section = False
            for line in output.splitlines():
                if "Benchmark Completed" in line:
                    benchmark_section = True
                    continue

                if benchmark_section:
                    if "ProcessingTime:" in line:
                        results["processing_time"] = float(
                            re.search(r"(\d+\.\d+)s", line).group(1)
                        )
                    elif "ProcessingSpeed:" in line:
                        results["processing_speed"] = float(
                            re.search(r"(\d+\.\d+)T/s", line).group(1)
                        )
                    elif "GenerationTime:" in line:
                        results["generation_time"] = float(
                            re.search(r"(\d+\.\d+)s", line).group(1)
                        )
                    elif "GenerationSpeed:" in line:
                        results["generation_speed"] = float(
                            re.search(r"(\d+\.\d+)T/s", line).group(1)
                        )
                    elif "TotalTime:" in line:
                        results["total_time"] = float(
                            re.search(r"(\d+\.\d+)s", line).group(1)
                        )
                        results["status"] = "success"

            results["memory_info"] = memory_info

        except Exception as e:
            results["parse_error"] = str(e)

        return results


# FastMCP Server Setup
mcp = FastMCP("KoboldCPP Benchmark Optimizer")
optimizer = BenchmarkOptimizer()


@mcp.tool()
async def analyze_model_and_system(model_path: str) -> Dict[str, Any]:
    """Analyze GGUF model file and extract metadata from remote system
        The purpose of this tool is to give you the information needed to run
        a benchmark. GPU layers and MOE layers should add up to the total layer amount.
        Compute the size of a layer by using the model weights size leaving 20% for context.
    
    Args:
        model_path (str): the path of huggingface URL to a set of gguf model weights.
    """
        
    try:
        result = await optimizer.setup_ssh_connection()
        model_info = await optimizer.analyze_model(model_path)
        system_info = await optimizer.detect_system_capabilities()
        return {"success": True, "model_info":model_info, "system_info": system_info}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def discover_system() -> Dict[str, Any]:
    """Discover remote system capabilities (VRAM, GPU count, etc.)"""
    try:
        result = await optimizer.detect_system_capabilities()
        return {"success": True, "system_info": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def setup_ssh() -> Dict[str, Any]:
    """Setup SSH connection to remote system"""
    try:
        result = await optimizer.setup_ssh_connection()
        return {"success": True, "connection_info": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def run_benchmark(
    model_path: str,
    gpulayers: int,
    moecpu: int = 0,
    contextsize: int = 2048,
    blasbatchsize: int = 512,
    #other_args: List[str] = None,
) -> Dict[str, Any]:
    """Run single benchmark with specified parameters
    
    Args:
        model_path: Path or huggingface URL to model file
        gpulayers: Number of GPU layers to use 
        moecpu: Number of MoE CPU layers (default 0)
        contextsize: Context size for benchmark (default 2048)
        blasbatchsize: BLAS batch size (-1,16,32,64,128,256,512,1024,2048) (default 512)
    """
    try:
        result = await optimizer.run_benchmark_via_ssh(
            model_path, gpulayers, moecpu, contextsize, blasbatchsize
        )
        return {"success": True, "experiment": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def get_optimization_status(model_path: str = None) -> Dict[str, Any]:
    """Get current optimization status and results"""
    try:
        if model_path:
            model_hash = optimizer.get_model_hash(model_path)
            experiments = optimizer.experiments_table.search(
                TinyQuery().model_hash == model_hash
            )
        else:
            experiments = optimizer.experiments_table.all()

        # Get best result
        best_experiment = None
        best_time = float("inf")

        for exp in experiments:
            if exp["results"].get("status") == "success":
                total_time = exp["results"].get("total_time", float("inf"))
                if total_time < best_time:
                    best_time = total_time
                    best_experiment = exp

        return {
            "success": True,
            "total_experiments": len(experiments),
            "successful_experiments": len(
                [e for e in experiments if e["results"].get("status") == "success"]
            ),
            "best_experiment": best_experiment,
            "all_experiments": experiments,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def reset_optimization(model_path: str = None) -> Dict[str, Any]:
    """Clear optimization results (optionally for specific model)"""
    try:
        if model_path:
            model_hash = optimizer.get_model_hash(model_path)
            removed = len(
                optimizer.experiments_table.search(TinyQuery().model_hash == model_hash)
            )
            optimizer.experiments_table.remove(TinyQuery().model_hash == model_hash)
        else:
            removed = len(optimizer.experiments_table)
            optimizer.experiments_table.truncate()

        return {"success": True, "removed_experiments": removed}

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run()