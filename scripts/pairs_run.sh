export CUDA_VISIBLE_DEVICES=0

python3 eval_dataset.py  --dataset='SummEval' \
                    --aspect='coherence' \
                    --eval_method='pairwise comparison' \
                    --eval_size=1600 \
                    --engine="microsoft/Phi-3-medium-4k-instruct" \
                    --with_input \
                    --confidence_beam \
                    --beam_size=5000 \
                    --prob_gap=0.2 \