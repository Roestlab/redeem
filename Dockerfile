# Use the official NVIDIA CUDA base image with CUDA 12.2
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libssl-dev \
    pkg-config \
    clang \
    libstdc++-12-dev \
    cmake \  
    git \    
    && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set the CUDA compute capability for the build process
# Tesla V100 has compute capability 7.0
ENV CUDA_COMPUTE_CAP=70

# Set the working directory
WORKDIR /app

# Copy the source code into the container
COPY . .

# Build the application with CUDA support
RUN cargo build --release --bin redeem --features cuda 

# Copy the binary into the PATH
RUN cp target/release/redeem /app/redeem

# Set the PATH environment variable
ENV PATH="/app:${PATH}"