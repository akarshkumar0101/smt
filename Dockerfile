FROM nvcr.io/nvidia/pytorch:26.01-py3

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /bin/

RUN apt-get update && apt-get install -y tmux && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --no-cache-dir --break-system-packages \
    numpy \
    pandas \
    xarray \
    matplotlib \
    seaborn \
    tqdm \
    einops \
    einop \
    tyro \
    python-dotenv \
    jupyterlab \
    ipywidgets \
    x_transformers \
    huggingface_hub \
    transformers \
    datasets \
    tiktoken \
    sentencepiece

WORKDIR $HOME
CMD ["/bin/bash"]
