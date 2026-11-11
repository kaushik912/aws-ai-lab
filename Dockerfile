# Lambda container image using AWS Lambda Web Adapter: runs plain uvicorn/FastAPI on Lambda,
# same image runs locally. Run `python scripts/ingest.py` first: ./index is baked in.
FROM python:3.12-slim AS build
WORKDIR /w
COPY pyproject.toml .
COPY app ./app
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
COPY --from=build /install /usr/local
WORKDIR /var/task
COPY app ./app
COPY index ./index
ENV PORT=8000 \
    AWS_LWA_INVOKE_MODE=response_stream \
    AWS_LWA_READINESS_CHECK_PATH=/healthz
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
