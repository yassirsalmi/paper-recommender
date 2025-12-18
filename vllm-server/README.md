# Building vLLM from Source on Mac

## Setup Python Environment
```bash
python3.12 -m venv vllm_env
source vllm_env/bin/activate
```

## Clone and Install vLLM
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install torch torchvision
```

## Install vLLM

the installation might fail with CUDA missing errors, use these environment variables before the installation to pass CUDA requirements:
```bash
export VLLM_TARGET_DEVICE=cpu
export VLLM_BUILD_WITH_CUDA=0
pip install -e .
```

## Run vLLM Server
```bash
vllm serve /path/to/your/model/ \
  --port 8099 \
  --served-model-name <model_name> \
  --max-model-len 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --chat-template /path/to/your/template.jinja
```