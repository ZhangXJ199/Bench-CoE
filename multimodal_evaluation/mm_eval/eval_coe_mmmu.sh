DATASET_DIR="/data/vlm/llava_data/eval/MMMU"

python -m eval_coe_mmmu \
    --model1_path /data/vlm/zxj/checkpoints/TinyLLaVA-Phi-2-SigLIP-3.1B \
    --model2_path /data/vlm/zxj/checkpoints/Bunny-v1_1-4B \
    --bert_model_path /data/vlm/zxj/checkpoints/subject_bert_mmmu \
    --image-folder $DATASET_DIR/all_images \
    --question-file $DATASET_DIR/anns_for_eval.json \
    --answers-file $DATASET_DIR/answers/coe.jsonl \

python convert_answer_to_mmmu.py \
    --answers-file $DATASET_DIR/answers/coe.jsonl \
    --answers-output $DATASET_DIR/answers/coe.json

cd $DATASET_DIR/eval

python main_eval_only.py --output_path $DATASET_DIR/answers/coe.json