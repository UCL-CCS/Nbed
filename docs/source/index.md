% Nbed documentation master file, created by
% sphinx-quickstart on Wed Feb 16 14:46:53 2022.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.
# Nbed

Projection-based embedding is a method for framenting a molecular system so that a computationally expensive method can be applied to an _active region_ of interest, while the _environment_ is simulated with a cheaper approxmimate method.

As a result, it is possible to obtain more accurate results than the cheaper method alone, while using significantly fewer computational resources than applying the expensive method to the whole system.

`nbed` is designed with application to quantum compting in mind, outputting a reduced dimension second quantised Hamiltonian which can be solved using any quantum simulation algorithm.

For more information, see [the example notebooks](Examples) and [our publications](Publications).


```{toctree}
:caption: 'Contents:'
:maxdepth: 1

examples
embed
driver
config
ham_builder
localizers
scf
utils
publications
```
