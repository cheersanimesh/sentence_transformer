FROM continuumio/miniconda3:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 1) build env in /tmp where it's always writable
WORKDIR /tmp/env
COPY environment.yml requirements.txt ./
RUN conda env create -f environment.yml \
    && conda clean -afy

# 2) put the new env on PATH
ENV PATH=/opt/conda/envs/sentence-transformer/bin:$PATH

# 3) now copy your app into /app
WORKDIR /app
COPY . .

CMD ["bash","-lc","python task1.py && python task2.py && python task4.py"]


# # Use the official Miniconda3 image
# FROM continuumio/miniconda3:latest

# # Don’t write .pyc files; unbuffered stdout/stderr
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1

# # Set working directory
# WORKDIR /app

# # Copy environment spec and pip requirements
# COPY environment.yml requirements.txt ./

# # Create the conda env and clean up
# RUN conda env create -f environment.yml \
#     && conda clean -afy

# # Ensure the new env is on PATH
# ENV PATH=/opt/conda/envs/sentence-transformer/bin:$PATH

# # Copy your application code
# COPY . .

# # On container start, run the three tasks in order
# # Use bash -lc so that the conda environment is fully initialized
# CMD ["bash", "-lc", "python task1.py && python task2.py && python task4.py"]