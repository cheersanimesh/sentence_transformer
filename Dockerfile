FROM continuumio/miniconda3:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


WORKDIR /app
COPY environment.yml requirements.txt ./
RUN conda env create -f environment.yml \
    && conda clean -afy


ENV PATH=/opt/conda/envs/sentence-transformer/bin:$PATH


WORKDIR /app
COPY . .

CMD ["bash","-lc","python task1.py && python task2.py && python task4.py"]
