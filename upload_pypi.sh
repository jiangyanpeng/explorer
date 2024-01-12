#! /bin/bash

# check setup is correct or not
python setup.py check

sudo rm -r build/
sudo rm -r dist/

# using twine instead
python setup.py sdist
twine upload dist/*
