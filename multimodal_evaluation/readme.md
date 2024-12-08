# Multimodal Project Documentation

## Test CoE model by lmms-eval

1. Install the environment based on lmms-eval and the selected model.

2. Train the router(Bert) using the MMMU dataset.

3. Place `coemodel.py` in the `.lmms_eval/models` directory and modify 'AVAILABLE_MODELS' list in `__init__.py`

4. Modify the paths for the models and the router in the code.

5. Run the command and select the corresponding task to test coemodel:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python3 -m accelerate.commands.launch \
      --num_processes=8 -m lmms_eval \
      --model coemodel \
      --model_args pretrained="None" \
      --tasks name --batch_size 1 \
      --log_samples --log_samples_suffix coemodel \
      --output_path ./logs/
   ```

Note: As long as the selected model and task are supported by lmm-eval, you only need to make the necessary modifications in `coemodel.py`.

## Test CoE model by yourself(case):

1. Install the environment based on [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) and [Bunny](https://github.com/BAAI-DCAI/Bunny), prepare the corresponding dataset.
```bash
# Create and activate environment
conda create -n bench-coe python=3.10 -y
conda activate bench-coe
cd /path/to/your/Bench-CoE/multimodal_evaluation/mm_eval/TinyLLaVA_Factory
pip install -e .
pip install xformers==0.0.20
```

2. Train the router(Bert) using the MMMU dataset.

3. Run the command to test coemodel on the MMMU dataset:
   ```Shell
   CUDA_VISIBLE_DEVICES=0 bash path/to/eval_coe_mmmu.sh
   ```

Note: When using this method, if you need to add a model, make modifications according to the loading method of the corresponding model.
