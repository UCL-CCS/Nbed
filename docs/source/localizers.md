# Localizers

Localization is performed in two independent steps.

The occupied-occupied block of the molecular orbital matrix is localized first,
before occupied environment orbitals are projected out.

After projection is performed, virtual orbitals can be localized using the embedded molecular orbitals.

## Occupied

```{eval-rst}
.. automodule:: nbed.localizers.occupied
   :members:
   :undoc-members:
   :show-inheritance:
```

## Virtual

```{eval-rst}
.. automodule:: nbed.localizers.virtual
   :members:
   :undoc-members:
   :show-inheritance:
```

## Localisation Data

Outputs from `OccupiedLocalizer.localize` are standardised as a dataclass, `LocalizedSystem`

```{eval-rst}
.. automodule:: nbed.localizers.system
   :members:
   :undoc-members:
   :show-inheritance:
```
