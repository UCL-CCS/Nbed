"""
Simulator class to run embedding with fragments
"""

from ..fragmentation.fragment_pair import FragmentPair

class Simulator:
    "Runs embedding simulations"

    def __init__(self, fragments: FragmentPair) -> None:
        self._frags = fragments

    def run(self) -> None:
        "Function to contain and run simulations."
        raise NotImplementedError