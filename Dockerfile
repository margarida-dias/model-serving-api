# base
FROM python:3.9 AS base

RUN mkdir -p /home/ffuser/app
COPY . /home/ffuser/app
WORKDIR /home/ffuser/app

RUN apt update -qq && \
	apt install --no-install-recommends -y make

RUN make install-dev

# builder
FROM base AS builder

RUN mkdir -p /home/ffuser/app
WORKDIR /home/ffuser/app

ENV PATH="/home/ffuser/.local/bin:$PATH"

COPY --chown=ffuser:ffuser --from=base /usr/bin/make /home/ffuser/app
COPY --chown=ffuser:ffuser --from=base /home/ffuser/app/Makefile /home/ffuser/app/
COPY --chown=ffuser:ffuser --from=base /home/ffuser/app/pyproject.toml /home/ffuser/app/
COPY --chown=ffuser:ffuser --from=base /home/ffuser/app/poetry.lock /home/ffuser/app/
COPY --chown=ffuser:ffuser --from=base /home/ffuser/app/servingapi /home/ffuser/app/servingapi
COPY --chown=ffuser:ffuser --from=base /home/ffuser/app/data /home/ffuser/app/data
COPY --chown=ffuser:ffuser --from=base /tmp/poetry-cache /tmp/poetry-cache

RUN ./make install
RUN rm -rf /home/ffuser/app/Makefile \
           /home/ffuser/app/poetry.lock \
           /home/ffuser/app/make \
           /tmp/poetry-cache

# production.
FROM builder AS main

USER ffuser

ENV PATH="/home/ffuser/app/.venv/bin:$PATH"

ENV MPLCONFIGDIR="/tmp"
ENV MLSERVER_MODEL_URI="/home/ffuser/app/data"
ENV PROMETHEUS_MULTIPROC_DIR="/tmp"

COPY --chown=ffuser:ffuser --from=builder /home/ffuser/app \
                                          /home/ffuser/app

ENTRYPOINT ["newrelic-admin", "run-program", "mlserver", "start", "app/servingapi"]
