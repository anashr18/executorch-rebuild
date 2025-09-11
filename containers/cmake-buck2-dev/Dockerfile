# Buck2 + CMake dev container (template copy)
FROM ubuntu:22.04

SHELL ["/bin/bash", "-lc"]

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_VERSION=3.20.0
ARG RUST_NIGHTLY=nightly-2025-06-20

# Install locations so all users can access Rust/Cargo and CMake
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/opt/cmake/bin:/usr/local/bin:/usr/bin:/bin

# Base build deps and useful tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates curl wget git gawk tree vim \
    build-essential pkg-config ninja-build ccache \
    clang clang-tidy clang-format lcov valgrind doxygen graphviz cppcheck \
    protobuf-compiler libprotobuf-dev libpqxx-dev \
    python3 python3-pip \
  && rm -rf /var/lib/apt/lists/*

# Install CMake (parametrized; matches project minimum requirement >= 3.19)
RUN cd /tmp && \
    mkdir -p /opt/cmake && \
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    chmod +x ./cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    ./cmake-${CMAKE_VERSION}-Linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    rm ./cmake-${CMAKE_VERSION}-Linux-x86_64.sh

# Install rustup system-wide and pin the nightly toolchain for Buck2
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain ${RUST_NIGHTLY}

# Build and install Buck2 from source (pinned nightly)
RUN cargo +${RUST_NIGHTLY} install --git https://github.com/facebook/buck2.git buck2 && \
    # reduce image size by clearing cargo caches
    rm -rf ${CARGO_HOME}/registry ${CARGO_HOME}/git

# Optional: create a non-root user for devcontainer usage
# ARG USERNAME=vscode
# ARG USER_UID=1000
# ARG USER_GID=$USER_UID
# RUN groupadd --gid $USER_GID $USERNAME && \
#     useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME && \
#     apt-get update && apt-get install --no-install-recommends -y sudo && \
#     echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
#     chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /work

