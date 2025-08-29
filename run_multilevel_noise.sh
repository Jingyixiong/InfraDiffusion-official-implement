#!/bin/bash

# Define the list of sigma_y values you want to test
# SIGMA_VALUES=(0 0.08 0.16 0.32)
SIGMA_VALUES=(0.08)

# Loop through each value in the list
for SIGMA in "${SIGMA_VALUES[@]}"
do
  echo "============================================="
  echo "Running inpainting with sigma_y = $SIGMA"
  echo "============================================="
  
  python main.py \
    --ni \
    --simplified \
    --config "infra.yml" \
    --deg "inpainting" \
    --sigma_y "$SIGMA"
done

echo "All experiments finished."