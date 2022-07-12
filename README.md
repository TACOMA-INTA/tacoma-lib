# TACOMA lib 
A python library for applying Machine Learning in CFD. This library is on active deveplopment and expect the API to change overtime.
At the moment, most of the classes and functions are centered on Reduce Order Modelling techniques (ROMs).

The authors of this library are:
* Jaime Bowen Varela (jbowvar(at)inta.es)
* Rodrigo Castellanos (rcasgar(at)inta.es)
* Alejandro Gorgues (gorguesva(at)inta.es)
## How to install
The official Pypi distribution can be downloaded from Pypi with:
```bash
pip install Tacoma-lib
```
## Documentation
The official documentation is under construction. There are couple of examples available to understand some of the inner workings of the library. 
## Modules of the library
The modules of the libray are divided into:
* rom: contains the Reduced Order Modelling classes
* interpolator: contains interpolars and redefyined classes
* metrics: contains clasess for metrics

There are a couple of more of modules that contains utils and stuff. This will surely change in the future. 
## Examples
The examples are contained in [here](https://github.com/jaimebw/tacoma_lib/tree/main/examples). The examples show some implemetations of the Tacoma lib for ROMs:
* [An example on how to use ROMs with Sklearn](https://github.com/TACOMA-INTA/tacoma-lib/blob/main/examples/rom_regression_example.py)
* [An example on how to do hyperoptimization with Sklearn and Optuna](https://github.com/TACOMA-INTA/tacoma-lib/blob/main/examples/rom_sklearn_optuna_example.py)
* [An example on how to use ROMs with PyTorch](https://github.com/TACOMA-INTA/tacoma-lib/blob/main/examples/rom_pytorch_example.py)

More examples to be added in the future
