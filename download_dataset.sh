#!/bin/bash
DIRECTORY=data
if [ ! -d "$DIRECTORY" ]; then
  echo Dir exist...
fi
mkdir data
cd data
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
unzip MPI-Sintel-complete.zip -d MPI-Sintel

mv MPI-Sintel ..

