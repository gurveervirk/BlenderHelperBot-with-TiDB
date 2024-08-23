# BlenderHelperBot

This is an _online RAG_ app, built using **TiDB vector store**, **llama-index** and **HuggingFace Inference API**, to help users with the **bpy** python package for **Blender** scripting.

TiDB vector store being used has already been fed **Blender 4.1 API docs**, for setting up the necessary context. The attached [`upload_to_tidb.py`](https://github.com/gurveervirk/BlenderHelperBot-with-TiDB/blob/main/upload_to_tidb.py) has been used for this purpose. 

## Requirements

1) Get a free [HuggingFace API read token](https://huggingface.co/settings/tokens).
2) Get access to [this Mistral model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3). This is being used as the underlying, instructed model.

## Known Issues

- Model may respond with some extra, unnecessary tokens / words, possibly due to it being a HuggingFace Inference API hosted model.
