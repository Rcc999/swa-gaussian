FROM python:3.8.20-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv
WORKDIR /code
# Before the rest of the handout, so editing solution.py does not reinstall.
COPY requirements.txt /code/requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt
ADD * /code/
ADD pytransform /code/pytransform
ENV LD_LIBRARY_PATH=/code/pytransform
CMD python -u checker_client.py
