# KAOGExp

Scientific research project.
Developed with Prof. Dr. João Roberto Bertini Jr.

## Dependencies

All the project dependencies can be found
under [requirements.txt](https://github.com/Anakin86708/kaogexp/blob/master/requirements.txt).

## Installation

`pip install -e kaogexp`

If you have some problems with the instalation of the `kaog` module, you can try it manualy following the steps
avaliable in [here](https://github.com/Anakin86708/kaog).

## Running

All the experiments and results are located in the [main](https://github.com/Anakin86708/kaogexp/tree/master/main)
directory.
All the scripts used in the paper in the following links:

- [Adult](https://github.com/Anakin86708/kaogexp/blob/master/main/carla_runs/adult/adult_multithread.py)
- [COMPAS](https://github.com/Anakin86708/kaogexp/blob/master/main/carla_runs/compas/compas_multithread.py)
- [Credit](https://github.com/Anakin86708/kaogexp/blob/master/main/carla_runs/credit/credit_multithread.py)

## Using

The KAOGExp can be used creating an object from `KAOGExp`. To explain one instance, you can use the method `explicar()`,
that requires some parameters like the method to use (`Counterfactual` or anything implementing `MethodAbstract`), the
data cleaner and the normalizer associated with the dataset and model.

--------
More documentation should be added later.

