# Configuration


Input data is validated against a Pydantic model, in the `NbedConfig` class. This is then passed to the `NbedDriver`.

## Example Configuration File

For the command-line interface to Nbed, you'll need to provide a path to a config file. This should contain a Json dictionary of input values.

```{eval-rst}
.. code-block:: json

   {
   "geometry":"1\\n\\nH\\t0.0\\t0.0\\t0.0",
   "n_active_atoms":1,
   "basis":"sto-3g",
   "xc_functional":"b3lyp",
   "projector":"mu",
   "localization":"spade",
   "convergence":1e-6,
   "charge":0,
   "spin":0,
   "unit":"angstrom",
   "symmetry":false,
   "mu_level_shift":1000000.0,
   "run_ccsd_emb":false,
   "run_fci_emb":false,
   "run_virtual_localization":true,
   "run_dft_in_dft":false,
   "n_mo_overwrite":[null,null],
   "max_ram_memory":4000,
   "occupied_threshold":0.95,
   "virtual_threshold":0.95,
   "max_shells":4,
   "init_huzinaga_rhf_with_mu":false,
   "max_hf_cycles":50,
   "max_dft_cycles":50,
   "force_unrestricted":false,
   "mm_coords":null,
   "mm_charges":null,
   "mm_radii":null
   }
```

## Config Model

```{eval-rst}
.. automodule:: nbed.config
   :members:
   :undoc-members:
   :show-inheritance:
```
