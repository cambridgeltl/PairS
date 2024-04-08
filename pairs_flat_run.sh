export CUDA_VISIBLE_DEVICES=1

python3 scripts/sorting.py  --dataset='hanna' \
                    --save_path='./hanna_compare.jsonl' \
                    --aspect='sensible' \
                    --eval_method='pairwise comparison' \
                    --eval_size=1600 \
                    --engine="meta-llama/Llama-2-7b-chat-hf" \
                    --confidence_beam \
                    --prob_gap=0.2 \