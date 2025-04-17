FROM continuumio/miniconda3

# Set the working directory
WORKDIR /rag-carbon

# Copy environment file
COPY environment.yml .

RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "faiss", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure faiss is installed:"
RUN python -c "import faiss"

# Copy faiss scripts
COPY faiss_lib.py .
COPY run_faiss.py .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "faiss", "python", "run_faiss.py"]
