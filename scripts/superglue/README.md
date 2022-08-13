# convert pre-trained superglue pytorch weights to onnx format

---

## dependencies

---

- python: 3x

-

```bash
git submodule update --init --recursive

python3 -m pip install -r requirements.txt
python3 -m pip install -r SuperGluePretrainedNetwork/requirements.txt
```

## :running: how to run

---

- export onnx weights

```
python3 convert_to_onnx.py
```
