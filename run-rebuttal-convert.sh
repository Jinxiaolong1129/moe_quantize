export PYTHONPATH=/data2/pzli/moe_quantize

python convert_openmoe_weight_to_hf.py \
  --input_dir="OrionZheng/openmoe-8b-800B" --output_dir="/data2/pzli/openmoe-checkpoints/openmoe-8b-800B-native-pt"

python convert_openmoe_weight_to_hf.py \
  --input_dir="OrionZheng/openmoe-8b-400B" --output_dir="/data2/pzli/openmoe-checkpoints/openmoe-8b-400B-native-pt"

