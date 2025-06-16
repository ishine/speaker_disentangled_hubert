#!/bin/sh

# setup
# conda create -y -n vghubert python=3.10.14 pip=24.0
# conda activate vghubert
# pip install -r requirements/requirements_for_vghubert.txt

if [ ! -d models/vg-hubert_3 ]
then
    echo downloading officially distributed VG-HuBERT models
    wget -P models -c https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/vg-hubert_3.tar
    tar -xvf models/vg-hubert_3.tar -C models
fi

python main_speech2unit.py syllable_segmentation --config configs/speech2unit/vghubert.yaml
python main_speech2unit.py quantize --config configs/speech2unit/vghubert.yaml
python main_speech2unit.py evaluate --config configs/speech2unit/vghubert.yaml