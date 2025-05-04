# Use an Ubuntu base image
FROM ubuntu:22.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set environment variables for Conda
ENV PATH="/opt/conda/bin:$PATH"
ENV CONDA_DEFAULT_ENV=octo

# Create and activate the conda environment
RUN conda create -n octo python=3.10
RUN echo "conda activate octo" >> ~/.bashrc

# Copy the Octo repository (assuming it's in the same directory as the Dockerfile)
COPY . /app

# Install Octo and its dependencies
RUN /opt/conda/envs/octo/bin/pip install -e .
RUN /opt/conda/envs/octo/bin/pip install -r requirements.txt

# Install JAX with CUDA support.  This is crucial for GPU acceleration.
RUN /opt/conda/envs/octo/bin/pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Set the entry point.  This is a placeholder; you'll need to change it
# to the actual command you want to run (e.g., a training script,
# a testing script, or a server).  You'll need to determine the
# correct command based on how you intend to use Octo.
#
# For example, if you have a script named 'run_octo.py' that you want to execute,
# the entrypoint would be:
#
# ENTRYPOINT ["/opt/conda/envs/octo/bin/python", "run_octo.py"]
#
# If you want to start a Jupyter Notebook server, it might be:
#
# ENTRYPOINT ["/opt/conda/envs/octo/bin/jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
#
# IMPORTANT:  Replace this with the *actual* command you want to run.
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/bin/activate octo && python scripts/finetune.py --config=scripts/configs/finetune_config.py:head_mlp_only,image_conditioned --config.pretrained_path=hf://rail-berkeley/octo-small-1.5 --config.save_dir=./checkpoints"]

# Expose the port if you are running a server (like a Jupyter Notebook).
# If you don't need to expose a port, you can remove this line.
# EXPOSE 8888
