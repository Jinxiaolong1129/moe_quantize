conda create -n gptq python=3.10 -y
conda activate gptq
pip install -r requirements.txt
cd lm-evaluation-harness && pip install -r requirements.txt && cd ..
