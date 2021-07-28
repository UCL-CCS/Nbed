"""
Entrypoint
"""
from pyscf import gto
from .utils.log import setup_logs
from .fragmentation import Fragmenter

if __name__ == "__main__":
    setup_logs()

    mol = 

    frag = Fragmenter