#!/bin/bash

set -exuo pipefail
trap "echo failed on line $LINENO" ERR

for i in {1995..2015}; do
    echo "year $i"
    python pts_parcels_io_beaching.py $i
done
