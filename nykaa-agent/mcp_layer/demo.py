"""Starts the server in one process and the client in another, prints the client's output."""
import os
import socket
import subprocess
import sys
import time

PORT = os.getenv("MCP_PORT", "8000")


def wait_for_port(port, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.3)
    return False


if __name__ == "__main__":
    server = subprocess.Popen([sys.executable, "-m", "mcp_layer.server"])
    try:
        assert wait_for_port(int(PORT)), "MCP server did not start"
        print(f"[demo] MCP server up at http://127.0.0.1:{PORT}/mcp (pid {server.pid}); launching separate client process")
        subprocess.run([sys.executable, "-m", "mcp_layer.client", "NYK-1003", "NYK-1048"], check=True)
    finally:
        server.terminate()
