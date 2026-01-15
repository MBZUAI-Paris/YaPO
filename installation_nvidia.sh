# create a new venv
echo "Creating conda environment 'yapo' with python 3.11.13"
conda create -n yapo python=3.11.13 -y

# activate the venv
echo "Initializing conda"
eval "$(conda shell.bash hook)"
conda deactivate
echo "Activating conda environment 'yapo'"
conda activate yapo

# up-to-date build tools
echo "Upgrading pip, setuptools, and wheel"
python -m pip install -U pip setuptools wheel

# install dependencies from PyPI
echo "Installing YaPO + evaluation dependencies from PyPI"
python -m pip install \
  absl-py==2.3.1 \
  accelerate==1.10.1 \
  aiohappyeyeballs==2.6.1 \
  aiohttp==3.12.15 \
  aiosignal==1.4.0 \
  amdsmi==6.3.0 \
  annotated-types==0.7.0 \
  anyio==4.10.0 \
  attrs==25.3.0 \
  bitsandbytes==0.42.0 \
  cachetools==6.2.1 \
  certifi==2025.8.3 \
  chardet==5.2.0 \
  charset-normalizer==3.4.3 \
  click==8.2.1 \
  colorama==0.4.6 \
  contourpy==1.3.3 \
  cycler==0.12.1 \
  data==0.4 \
  dataproperty==1.1.0 \
  datasets==4.1.0 \
  decorator==5.2.1 \
  deepspeed==0.17.5 \
  dill==0.3.8 \
  distro==1.9.0 \
  docstring-parser==0.17.0 \
  einops==0.8.1 \
  evaluate==0.4.6 \
  fastapi==0.116.1 \
  filelock==3.13.1 \
  fonttools==4.60.0 \
  frozenlist==1.7.0 \
  fschat==0.2.36 \
  fsspec==2023.10.0 \
  funcsigs==1.0.2 \
  future==1.0.0 \
  gitdb==4.0.12 \
  gitpython==3.1.45 \
  google-ai-generativelanguage==0.6.15 \
  google-api-core==2.28.1 \
  google-api-python-client==2.187.0 \
  google-auth==2.43.0 \
  google-auth-httplib2==0.2.1 \
  google-generativeai==0.8.5 \
  googleapis-common-protos==1.72.0 \
  grpcio==1.76.0 \
  grpcio-status==1.71.2 \
  h11==0.16.0 \
  hf-xet==1.2.0 \
  hjson==3.1.0 \
  httpcore==1.0.9 \
  httplib2==0.31.0 \
  httpx==0.28.1 \
  huggingface-hub==0.36.0 \
  idna==3.10 \
  jinja2==3.1.4 \
  joblib==1.5.3 \
  jsonlines==4.0.0 \
  kiwisolver==1.4.9 \
  latex==0.7.0 \
  latex2mathml==3.78.1 \
  lm-eval==0.4.9.2 \
  lxml==6.0.2 \
  markdown-it-py==4.0.0 \
  markdown2==2.5.4 \
  markupsafe==2.1.5 \
  matplotlib==3.8.4 \
  mbstrdecoder==1.1.4 \
  mdurl==0.1.2 \
  more-itertools==10.8.0 \
  mpmath==1.3.0 \
  msgpack==1.1.1 \
  multidict==6.6.4 \
  multiprocess==0.70.16 \
  networkx==3.3 \
  nh3==0.3.0 \
  ninja==1.13.0 \
  nltk==3.9.2 \
  numpy==1.26.4 \
  openai==1.12.0 \
  packaging==25.0 \
  pandas==2.3.2 \
  pathvalidate==3.3.1 \
  peft==0.8.2 \
  pillow==11.0.0 \
  platformdirs==4.4.0 \
  portalocker==3.2.0 \
  prompt-toolkit==3.0.52 \
  propcache==0.3.2 \
  proto-plus==1.26.1 \
  protobuf==5.29.5 \
  psutil==7.0.0 \
  py-cpuinfo==9.0.0 \
  pyarrow==21.0.0 \
  pyasn1==0.6.1 \
  pyasn1-modules==0.4.2 \
  pydantic==2.11.9 \
  pydantic-core==2.33.2 \
  pygments==2.19.2 \
  pyparsing==3.2.5 \
  pytablewriter==1.2.1 \
  python-dateutil==2.9.0.post0 \
  python-dotenv==1.1.1 \
  pytz==2025.2 \
  pyyaml==6.0.2 \
  regex==2025.9.1 \
  requests==2.32.5 \
  rich==14.1.0 \
  rouge-score==0.1.2 \
  rsa==4.9.1 \
  sacrebleu==2.5.1 \
  safetensors==0.6.2 \
  scikit-learn==1.8.0 \
  scipy==1.16.2 \
  seaborn==0.13.2 \
  sentry-sdk==2.38.0 \
  setuptools==80.9.0 \
  shellingham==1.5.4 \
  shortuuid==1.0.13 \
  shtab==1.7.2 \
  shutilwhich==1.1.0 \
  six==1.17.0 \
  smmap==5.0.2 \
  sniffio==1.3.1 \
  sqlitedict==2.1.0 \
  starlette==0.47.3 \
  svgwrite==1.4.3 \
  sympy==1.13.3 \
  tabledata==1.3.4 \
  tabulate==0.9.0 \
  tcolorpy==0.1.7 \
  tempdir==0.7.1 \
  threadpoolctl==3.6.0 \
  tiktoken==0.11.0 \
  tikzplotlib==0.10.1 \
  tokenizers==0.22.1 \
  tqdm==4.67.1 \
  transformers==4.57.1 \
  typepy==1.3.4 \
  typer-slim==0.20.0 \
  typing-extensions==4.12.2 \
  typing-inspection==0.4.1 \
  tyro==0.7.2 \
  tzdata==2025.2 \
  uritemplate==4.2.0 \
  urllib3==2.5.0 \
  uvicorn==0.35.0 \
  wandb==0.21.4 \
  wavedrom==2.0.3.post3 \
  wcwidth==0.2.13 \
  webcolors==24.11.1 \
  word2number==1.1 \
  xxhash==3.5.0 \
  yarl==1.20.1 \
  zstandard==0.25.0

# (Optional but useful)
echo "Final check of installed packages"
python -m pip check
