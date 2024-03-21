# Start from an ARM64 compatible base image
FROM arm64v8/ubuntu:20.04

# Set environment variables to avoid user interaction during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip3 install --upgrade pip && pip3 install wheel

# Install TensorFlow for ARM64
# Replace '2.x' with the specific version of TensorFlow you need, if necessary
RUN pip3 install tensorflow-aarch64

# Additional Python packages for data science and ML (optional)
RUN pip3 install numpy pandas matplotlib scikit-learn jupyter

# Set up a working directory
WORKDIR /workspace

# Expose port for Jupyter (optional)
EXPOSE 8888

# Start Jupyter Notebook (optional)
# You can change this to simply start a shell by replacing it with /bin/bash
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]

