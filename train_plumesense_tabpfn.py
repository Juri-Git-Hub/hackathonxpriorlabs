"""Compatibility entry point for the supplied classifier script."""
import subprocess
import sys
if __name__ == '__main__':
    subprocess.run([sys.executable,'run.py','evaluate','--dataset','original','--backend','tabpfn',*sys.argv[1:]],check=True)
