#!/bin/bash

rm -rf venv_momogp/
virtualenv -p python3 ./venv_momogp
source ./venv_momogp/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

