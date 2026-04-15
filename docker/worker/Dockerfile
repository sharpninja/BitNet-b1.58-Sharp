# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────
#  BitNetSharp.Distributed.Worker — CPU-only training worker container.
#
#  Build from the repository root so the build context can reach the
#  BitNetSharp.Core source that the worker project references:
#
#    docker build -t ghcr.io/sharpninja/bitnetsharp-worker:dev \
#                 -f docker/worker/Dockerfile .
#
#  At runtime the worker reads configuration from environment variables
#  (see src/BitNetSharp.Distributed.Worker/WorkerConfig.cs). The absolute
#  minimum wiring is:
#
#    BITNET_COORDINATOR_URL   public https URL of the coordinator (e.g. an
#                             ngrok reserved domain pointing at PAYTON-DESKTOP)
#    BITNET_ENROLLMENT_KEY    pre-shared secret for worker registration
#
#  Everything else has safe defaults baked into WorkerConfig and into this
#  Dockerfile's ENV block below.
# ─────────────────────────────────────────────────────────────────────────

# ── build stage ──────────────────────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
ARG TARGETARCH
WORKDIR /src

# Copy SDK pin + shared build props first so docker layer cache works.
COPY global.json Directory.Build.props ./

# Bring in only the projects the worker needs. BitNetSharp.Core supplies
# the ternary training primitives; the worker project is the Exe target.
COPY src/BitNetSharp.Core             ./src/BitNetSharp.Core
COPY src/BitNetSharp.Distributed.Worker ./src/BitNetSharp.Distributed.Worker

RUN dotnet restore src/BitNetSharp.Distributed.Worker/BitNetSharp.Distributed.Worker.csproj

RUN dotnet publish src/BitNetSharp.Distributed.Worker/BitNetSharp.Distributed.Worker.csproj \
      -c Release \
      -f net10.0 \
      -o /publish \
      --no-restore \
      --no-self-contained \
      /p:UseAppHost=false

# ── runtime stage ────────────────────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/runtime:10.0 AS runtime

LABEL org.opencontainers.image.source="https://github.com/sharpninja/BitNet-b1.58-Sharp"
LABEL org.opencontainers.image.title="BitNetSharp.Distributed.Worker"
LABEL org.opencontainers.image.description="CPU-only BitNet b1.58 distributed training worker (integer-accumulated ternary matmul)"
LABEL org.opencontainers.image.licenses="MIT"

# Create a non-root user and a working directory. Writes are confined to
# /work/cache and /tmp.
RUN groupadd --system --gid 10001 bitnet \
 && useradd  --system --uid 10001 --gid 10001 --home /work --shell /bin/false bitnet \
 && mkdir -p /work/cache \
 && chown -R bitnet:bitnet /work

WORKDIR /work
COPY --from=build --chown=bitnet:bitnet /publish ./

USER bitnet

# GC tuning + conservative .NET defaults suitable for a multi-core CPU
# training worker. Server GC keeps throughput high; tiered PGO speeds up
# the benchmark and the training hot path within the first few minutes.
ENV DOTNET_gcServer=1 \
    DOTNET_TieredPGO=1 \
    DOTNET_ReadyToRun=1 \
    DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 \
    BITNET_HEALTH_BEACON=/tmp/bitnet-worker-alive \
    BITNET_HEARTBEAT_SECONDS=30 \
    BITNET_SHUTDOWN_SECONDS=30 \
    BITNET_LOG_LEVEL=info

# Docker healthcheck: the worker touches /tmp/bitnet-worker-alive every
# heartbeat cycle. If the mtime falls more than two minutes behind, the
# worker is declared unhealthy and the orchestrator can replace it.
# The start-period is long enough for the startup BenchmarkDotNet
# calibration pass to complete on slow hardware before health checks
# begin running.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD test -f /tmp/bitnet-worker-alive && \
        [ $(( $(date +%s) - $(stat -c %Y /tmp/bitnet-worker-alive) )) -lt 120 ] \
        || exit 1

ENTRYPOINT ["dotnet", "BitNetSharp.Distributed.Worker.dll"]
