On the original repo the dependencies in requirements.txt have been updated to work with Python 3.12 (almost) even though the readme says it works with Python 3.10.

To make it work fully we updated the numpy version to 1.26.4.

To install it you can use `uv`:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then setup virtual env with Python 3.12:

```sh
uv venv -p 3.12
```

Then install the dependencies:

```sh
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

To run the inference with bounding box control (bounding box proportions control):

> This will run on hardcoded input data from [`demos/bbox/data.json`](demos/bbox/data.json) and [`demos/bbox/storage_rack1.png`](demos/bbox/storage_rack1.png)

```sh
python3 inference.py --control_type bbox
```

To run the inference with point cloud control:

> This will run on original demo data from Tencent. To run on Robotec data we need to downsample the point clouds to a couple thousand points.

```sh
python3 inference.py --control_type point
```

