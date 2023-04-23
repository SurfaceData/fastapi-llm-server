# FastAPI LLM Server

This is a bare bones FastAPI based Large Language Model Server.  It's meant for
local prototyping and deployment on AWS.

```sh
DOCKER_BUILDKIT=1 docker build -t surface-data/llm-server-gpu -f Dockerfile.gpu .
```

## Citations

This server includes support for
*  [Stable LM](https://github.com/Stability-AI/StableLM) produced by
   [Stability AI](https://stability.ai/) under the [CC
BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.
*  [ChatRWKV](https://github.com/BlinkDL/ChatRWKV), an RNN based LLM published with the Apache 2.0 license.
*  Most [Huggingface](https://huggingface.co/) models.
