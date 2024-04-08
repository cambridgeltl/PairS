export CUDA_VISIBLE_DEVICES=1

python3 scripts/eval_dataset.py  --dataset='newsroom' \
                    --aspect='coherence' \
                    --eval_method='pairwise comparison' \
                    --eval_size=1600 \
                    --engine="gpt-3.5-turbo" \
                    --with_input \
                    --confidence_beam \
                    --beam_size=5000 \
                    --prob_gap=0.2 \