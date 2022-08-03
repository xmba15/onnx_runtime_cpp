# convert pre-trained superpoint pytorch weights to onnx format

---

## dependencies

---

- python: 3x

-

```bash
python3 -m pip install -r requirements.txt
```

## :running: how to run

---

- update submodule

```bash
git submodule update --init --recursive
```

- export onnx weights

```
python3 convert_to_onnx.py
```
