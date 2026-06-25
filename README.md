<div align="center" style="line-height: 1;">
<h1>CoffeeBench: Benchmarking Long-Horizon LLM Agents in Heterogeneous Multi-Agent Economies</h1>


  |
  <a href="https://arxiv.org/abs/2606.16613" target="_blank">📄 Paper</a>
  &nbsp;|
  <a href="https://pub.sakana.ai/coffeebench/index.html" target="_blank">📝 Blog</a>
  &nbsp;|
    <a href="https://pub.sakana.ai/coffeebench/trajectories.html" target="_blank">🔍 Trajectories</a>
  &nbsp;|

  <br/>

<img src="./assets/cumulative_net_income.gif" width="80%"/>
</div>


## Overview

CoffeeBench is a benchmark for evaluating how much net income an LLM agent can generate as a coffee roaster over 90 days in a multi-agent economy with two farmers, two roasters, and two retailers.

<figure>
  <img src="./assets/CoffeeBench.svg" alt="Overview of CoffeeBench" style="width: 100%">
  <figcaption>Overview of CoffeeBench.</figcaption>
</figure>


## How to run

Install dependencies:
```bash
uv sync
```

Create `.env` and set your model-provider API keys (see `.env.example`):

```
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-..."
GEMINI_API_KEY="AI..."
OPENROUTER_API_KEY="sk-..."
```

Run a single simulation:

> NOTE: it takes over $200 in API costs and over 5 hours to run.
```bash
# Single 90-day run (Sonnet driving roaster_A; all other firms on Sonnet).
uv run python -m coffeebench.main \
    --config experiments/roaster_focal_sonnet.toml --seed 0
```




Monitor the simulation in real time with the web dashboard:
```bash
# Live web dashboard.
uv run streamlit run coffeebench/web.py
```

<figure>
  <img src="./assets/viewer.png" alt="Viewer of CoffeeBench" style="width: 100%">
  <figcaption>Screen shot of the web viewer.</figcaption>
</figure>


Run the full battery of experiments:

```bash
# Sweep the production matrix: 5 focal models × 3 seeds.
for tag in haiku sonnet opus gpt gemini; do
  for seed in 0 1 2; do
    uv run python -m coffeebench.main \
        --config experiments/roaster_focal_${tag}.toml --seed ${seed}
  done
done
```

## Citation
If you find our work interesting, please consider citing our paper:
```bibtex
@misc{sugiura2026coffeebenchbenchmarkinglonghorizonllm,
      title={CoffeeBench: Benchmarking Long-Horizon LLM Agents in Heterogeneous Multi-Agent Economies},
      author={Issa Sugiura and Daichi Hattori and Kazuo Araragi and Keita Ogawa and Shota Onose and Taro Makino and Teppei Usuki and Takashi Ishida},
      year={2026},
      eprint={2606.16613},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.16613},
}
```
