#!/bin/sh

git clone https://github.com/cheoljun95/sdhubert.git src/sdhubert
git clone https://github.com/jasonppy/syllable-discovery.git src/vghubert
git clone https://github.com/Berkeley-Speech-Group/sylber.git src/sylber

cd src/sdhubert
git checkout ecb6469
cd -

cd src/sdhubert/mincut
python setup.py build_ext --inplace
cd -

patch src/sdhubert/extract_segments.py src/patch/sdhubert_extract_segments.patch
patch src/sdhubert/utils/misc.py src/patch/sdhubert_utils_misc.patch