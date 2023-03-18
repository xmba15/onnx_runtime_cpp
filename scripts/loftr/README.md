# convert pre-trained loftr pytorch weights to onnx format

---

## dependencies

---

- python: 3x

-

```bash
git submodule update --init --recursive

python3 -m pip install -r requirements.txt
```

## :running: how to run

---

- download [LoFTR](https://github.com/zju3dv/LoFTR) weights indoor_ds_new.ckpt from [HERE](https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp)

- export onnx weights

```
python3 convert_to_onnx.py --model_path /path/to/indoor_ds_new.ckpt
```

## Note

- The LoFTR's [latest commit](b4ee7eb0359d0062e794c99f73e27639d7c7ac9f) seems to be only compatible with the new weights (Ref: https://github.com/zju3dv/LoFTR/issues/48). Hence, this onnx cpp application is only compatible with _indoor_ds_new.ckpt_ weights.
