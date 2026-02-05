<div align="center">

# rLLM

<div>
🚀 Reinforcement Learning for Language Agents🌟
</div>
</div>
<div>
<br>

</div>

rLLM is an open-source framework for post-training language agents via reinforcement learning. With rLLM, you can easily build your custom agents and environments, train them with reinforcement learning, and deploy them for real-world workloads.

## EMA-PG: Math Experiments

Our scripts can be found in `examples/deepscaler`.

## Getting Started 🎯

rLLM requires `Python >= 3.10` (`3.11` is needed if using `tinker`). You can install it either directly via pip or build from source.

There are three ways that you can install rLLM:

### Approach A: Direct Installation

```bash
uv pip install "rllm[verl] @ git+https://github.com/rllm-org/rllm.git"
```

_(or replace the `verl` above for `tinker` to install with tinker backend, see below for more details)_

### Approach B: Building from Source with `uv`

**Step 1: Clone and Setup Environment**

```bash
# Clone the repository
git clone https://github.com/rllm-org/rllm.git
cd rllm

# Create an uv environment
uv venv --python 3.11
source .venv/bin/activate
```

**Step 2: Install rLLM with Training Backend**

rLLM supports two training backends: `verl` and `tinker`. Choose one based on your needs.

_**Option I:** Using `verl` as Training Backend_

```bash
uv pip install -e .[verl] 
```

_**Option II:** Using `tinker` as Training Backend_

```bash
# can add --torch-backend=cpu to train on CPU-only machines
uv pip install -e .[tinker] 
```

### Approach C: Installation with Docker 🐳

For a containerized setup, you can use Docker:

```bash
# Build the Docker image
docker build -t rllm .

# Create and start the container
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/rllm -v /tmp:/tmp --name rllm-container rllm sleep infinity
docker start rllm-container

# Enter the container
docker exec -it rllm-container bash
```

For more detailed installation guide, including using `sglang` for `verl` backend, please refer to our [documentation](https://rllm-project.readthedocs.io/en/latest/getting-started/installation).

## Acknowledgements
Our work is done as part of [Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu/). The rLLM team is generously supported by grants from [Laude Institute](https://www.laude.org/), [AWS](https://aws.amazon.com/), [Hyperbolic](https://www.hyperbolic.ai/), [Fireworks AI](https://fireworks.ai/), and [Modal](https://modal.com/). We pay special thanks to [Together AI](https://www.together.ai/) for the research partnership and compute support. 

## Citation
```bibtex
@misc{rllm2025,
  title={rLLM: A Framework for Post-Training Language Agents},
  author={Sijun Tan and Michael Luo and Colin Cai and Tarun Venkat and Kyle Montgomery and Aaron Hao and Tianhao Wu and Arnav Balyan and Manan Roongta and Chenguang Wang and Li Erran Li and Raluca Ada Popa and Ion Stoica},
  year={2025},
  howpublished={\url{https://pretty-radio-b75.notion.site/rLLM-A-Framework-for-Post-Training-Language-Agents-21b81902c146819db63cd98a54ba5f31}},
  note={Notion Blog}
  year={2025}
}
```

You may also cite our prior work [DeepScaleR](https://scholar.googleusercontent.com/scholar.bib?q=info:PrmBADk39GwJ:scholar.google.com/&output=citation&scisdr=CgIJFx-xEMCQ6zOgcuI:AAZF9b8AAAAAaPCmauIfzg8Rm9ImNYDad0uPUK8&scisig=AAZF9b8AAAAAaPCmahXsNqb1jTQBw2iPfw2vm9g&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1).
