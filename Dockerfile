FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch first to avoid pulling the ~2 GB CUDA build
# from PyPI's default index. Everything else in requirements.txt is then
# installed in a second pass; pip skips torch because it is already present.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN N_SCENARIOS=2000 python data/generate_data.py

EXPOSE 8080

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
