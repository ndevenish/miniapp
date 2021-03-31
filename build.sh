#!/bin/bash

set -e

# Takes a path to the dials build path first
if [[ -z $1 && ! -f .dials_build ]]; then
    echo "Usage: build.sh [/path/to/dials/build]"
    echo "    Path to DIALS build required first time."
    exit 1
fi
if [[ -f .dials_build ]]; then
    DIALS_BUILD="$(cat .dials_build)"
    export DIALS_BUILD
else
    DIALS_BUILD="$1"
fi

if [[ ! -d $DIALS_BUILD || ! -f "$DIALS_BUILD/libtbx_env" ]]; then
    echo "Error: $DIALS_BUILD does not look like a dials build dir?"
    exit 1
fi
echo "$DIALS_BUILD" > .dials_build

eval "$(conda shell.bash hook)"

if [[ ! -d ENV ]]; then
    conda create -yp ENV boost-cpp benchmark gtest cmake
fi

conda activate ENV/
mkdir -p baseline/_build
cd baseline/_build
cmake .. -DDIALS_BUILD="$DIALS_BUILD"
make
echo "h5read and baseline should be in baseline/_build"

