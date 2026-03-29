# Stage 1: Build external dependencies (impg + AGC)
FROM rust:1.75-bookworm AS deps

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake git zlib1g-dev make gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Build AGC (C++ project, uses make)
# TODO (2026-03-28): pin to a specific release tag or commit hash for reproducibility
RUN git clone --depth 1 --branch v3.2 https://github.com/refresh-bio/agc.git /opt/agc && \
    cd /opt/agc && \
    make -j"$(nproc)"

# Build impg (Rust project)
# TODO (2026-03-28): pin to a specific release tag or commit hash for reproducibility
RUN git clone --depth 1 --branch v0.2.3 https://github.com/pangenome/impg.git /opt/impg && \
    cd /opt/impg && \
    cargo build --release

# Stage 2: Build impopk workspace
FROM rust:1.75-bookworm AS builder

COPY Cargo.toml Cargo.lock /build/
COPY src/ /build/src/

WORKDIR /build
RUN cargo build --release

# Stage 3: Minimal runtime image
FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        zlib1g wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# External tool binaries
COPY --from=deps /opt/agc/bin/agc /usr/local/bin/
COPY --from=deps /opt/impg/target/release/impg /usr/local/bin/

# impopk binaries
COPY --from=builder /build/target/release/ibs /usr/local/bin/
COPY --from=builder /build/target/release/ibs-from-paf /usr/local/bin/
COPY --from=builder /build/target/release/ibs-from-tpa /usr/local/bin/
COPY --from=builder /build/target/release/tpa-spatial-index /usr/local/bin/
COPY --from=builder /build/target/release/tpa-validate /usr/local/bin/
COPY --from=builder /build/target/release/ibd /usr/local/bin/
COPY --from=builder /build/target/release/ibd-validate /usr/local/bin/
COPY --from=builder /build/target/release/ancestry /usr/local/bin/
COPY --from=builder /build/target/release/jacquard /usr/local/bin/

# Bundled data (sample lists and genetic maps)
COPY data/samples/ /data/samples/
COPY data/genetic_maps/ /data/genetic_maps/
RUN chmod -R a+r /data/

# Run as non-root user
RUN useradd -m -s /bin/bash impopk
USER impopk
WORKDIR /home/impopk

ENTRYPOINT []
CMD ["bash"]
