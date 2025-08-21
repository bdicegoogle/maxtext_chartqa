FROM us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.7.0-rev1

# Set working directory
WORKDIR /workspace

# Clone MaxText repository
RUN git clone https://github.com/AI-Hypercomputer/maxtext.git

# Change to maxtext directory and install dependencies
WORKDIR /workspace/maxtext

RUN pip install psutil jaxtyping Pillow omegaconf transformers pathwaysutils && \
    pip install git+https://github.com/google/qwix && \
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install git+https://github.com/AI-Hypercomputer/JetStream.git

RUN apt update && apt install vim -y
COPY application_default_credentials.json /tmp/keys.json 

# Set working directory
WORKDIR /workspace

RUN git clone https://github.com/google/JetStream.git
WORKDIR /workspace/JetStream
RUN pip install -e .


# Create a startup script
RUN echo '#!/bin/bash\n\
set -ex\n\
export idx=$(date +%Y-%m-%d-%H-%M)\n\
echo "Environment setup complete. Ready for Gemma3 training."\n\
exec "$@"' > /workspace/startup.sh && \
    chmod +x /workspace/startup.sh

WORKDIR /workspace/maxtext
    

ENTRYPOINT ["/workspace/startup.sh"]
CMD ["/bin/bash"]
