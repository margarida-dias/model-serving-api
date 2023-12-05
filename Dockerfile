# Use a single multi-stage build
FROM python:3.9 AS builder

# Set up the working directory
RUN mkdir -p /app/
WORKDIR /app/

# Copy necessary files
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y make

COPY Makefile pyproject.toml /app/

RUN make install
RUN rm -rf /app/Makefile \
           /app/poetry.lock \
           /app/make \
           /tmp/poetry-cache

COPY . .

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV MLSERVER_MODEL_URI="/app/data"
ENV PROMETHEUS_MULTIPROC_DIR="/tmp"

# Set up a non-root user
USER root

# Entry point for the container
CMD mlserver start .
#ENTRYPOINT ["newrelic-admin", "run-program", "mlserver", "start", "app/servingapi"]


