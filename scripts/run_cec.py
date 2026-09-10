"""Entry point for the CEC2017 study.

Kept separate so the worker functions live in an ordinary imported module.
Running the study module itself with -m makes them members of __main__, which
Windows' spawn start method re-imports in every worker -- an unreliable path
that showed up here as intermittent un-serialize and worker-death errors.
"""
from experiments.cec2017_study import main

if __name__ == "__main__":
    main()
