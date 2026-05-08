# Learning to Explore: Scaling Agentic Reasoning via Exploration-Aware Policy Optimization
<p align="center">
&nbsp&nbsp🌐 <a href="https://xingyuan-project.github.io/m2cl.github.io/">Website</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href=>arXiv</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://huggingface.co/hansenhua/EAPO-ICML26">Model</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/hansenhua/EAPO-ICML26">Hugging Face</a>&nbsp&nbsp
</p>

<p align="center">
  <img src="introduction.png" alt="EAPO">
</p>

## 📢 Updates
<!-- - [2026-3-14] We publish the project page.
- [2026-2-5] We publish the paper on huggingface.
- [2026-2-4] We open-source the code for training.
- [2026-2-3] We publish the paper on arxiv. -->
- [2026-5-1] This paper was accepted by ICML'26

## 🔨 TODO
- [ ] Polish the codebase.
- [ ] Merge with the latest verl version.
- [ ] Release the model checkpoint.

## 🚀 Quick Start

This guide provides instructions for setting up the EAPO, including execution scripts for inference and training.

### 1. Preparation

#### Download Code

Download the code from Github.
```bash
git clone https://github.com/HansenHua/EAPO-ICML26.git
cd EAPO-ICML26
```

#### Prepare Model Checkpoints

Download the model using the HuggingFace CLI. Replace `<your local path>` with your actual directory.

```bash
huggingface-cli download model_path --local-dir <your local path>
```

### 2. Env Initialization

Initialize the python environment on your **GPU Machine**.

#### Install dependent packages

```bash
conda create -n your_env python=3.9
conda activate your-env
pip install -r requirements.txt
```

### 3. Execution

```bash
python main.py
```
The code allows for 
```
--dataset The name of the dataset.
--method The name of the method.
--model The backbone model name.
--model_path Path to the model checkpoint.
--generator_path Path to the context generator.
--num Number of agents.
--max_rounds Maximum number of debating rounds.
--seed Random seed.
--n Number of chat completion candidates.
--temperature Sampling temperature, in range [0, 2].
--alpha Alpha coefficient.
--beta Beta coefficient.
--max_completion_tokens Maximum number of completion tokens.
--top_k Top-k combinations selected.
--contribution_threshold Threshold for agent contribution filtering.
--process_num Number of parallel processes.
--train_rounds Maximum number of training rounds.
```

## 📝 Citation
If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝 :)

```bibtex

```
