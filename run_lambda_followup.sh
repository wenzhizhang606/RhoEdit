#!/bin/bash
# First chunk only: finish λ=0.1 on zsre+wiki, then λ=10 on all three datasets.
set -euo pipefail
cd /home/zwz/01_safeedit
sed -i 's/\r$//' run_0.sh run_first_chunk_hp.sh
chmod +x run_0.sh run_first_chunk_hp.sh

echo "[followup] $(date) phase1 first-chunk λ=0.1 zsre+wiki"
bash run_0.sh 1
echo "[followup] $(date) phase1 done; phase2 first-chunk λ=10 counterfact+zsre+wiki"
bash run_0.sh 2
echo "[followup] $(date) all done"
