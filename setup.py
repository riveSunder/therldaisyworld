from setuptools import setup

setup(name="daisy",\
        packages = ["daisy"],\
        version = "0.0",\
        description = "The RL DaisyWorld", \
        install_requires = ["numpy==1.24.2",\
                "matplotlib==3.7.0",\
                "mpi4py==3.1.5"] \
        )

    
