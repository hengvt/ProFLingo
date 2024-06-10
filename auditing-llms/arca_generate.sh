for i in {0..49}; do
    python reverse_experiment.py --save_every 10 --n_trials 1 --arca_iters 256 --arca_batch_size 16 --prompt_length 32 --lam_perp 0 --label your-file-label --filename questions.csv --opts_to_run arca --model_id llama2 --inpt_tok_constraint ascii --index $i
done