#!/bin/bash

set -e

ENV_VERSION=1

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

INSTALL_ENV=""
if [[ -d ENV ]]; then
    if [[ ! -f ENV/.miniapp_version || ! "$ENV_VERSION" == "$(cat ENV/.miniapp_version)" ]]; then
        echo "Environment incomplete or from previous version; reinstalling"
        INSTALL_ENV="yes"
    fi
fi

packages=(boost-cpp benchmark gtest cmake hdf5)

if [[ ! -d ENV ]]; then
    module load mamba
    mamba create -yp ENV "${packages[@]}"
    echo "$ENV_VERSION" > ENV/.miniapp_version
elif [[ $INSTALL_ENV == "yes" ]]; then
    module load mamba
    conda activate ENV/
    mamba install "${packages[@]}"
    echo "$ENV_VERSION" > ENV/.miniapp_version
fi

conda activate ENV/
mkdir -p baseline/_build
cd baseline/_build
cmake .. -DDIALS_BUILD="$DIALS_BUILD"
make
echo "h5read and baseline should be in baseline/_build"

