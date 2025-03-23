#!/bin/bash
session="create_esm_dataset"

script="./create_esm_dataset_hdf5.py"

tmux new-session -d -s $session

for frac in {0..499}; do
    frac_str=$(printf "%03d" "$frac")
    script_name=$(basename "$script" .py)
    win_name="${script_name}_${frac_str}"
    tmux new-window -t "$session":"$(( frac + 2 ))" -n "$win_name"
    tmux send-keys -t "$session":"$win_name" "mamba activate esmjax" Enter
    tmux send-keys -t "$session":"$win_name" "python $script $frac_str" Enter
done

tmux attach-session -t $session