import gc
import torch

from pynvml import *
from huggingface_hub import hf_hub_download


def create_raven_model():
    ctx_limit = 1024
    title = "RWKV-4-Raven-7B-v7-Eng-20230404-ctx4096"
    strategy = "cuda fp16i8 *8 -> cuda fp16"
    tokenize_path = "data/20B_tokenizer.json"
    repo_id = "BlinkDL/rwkv-4-raven"

    nvmlInit()
    gpu_h = nvmlDeviceGetHandleByIndex(0)

    os.environ["RWKV_JIT_ON"] = "1"
    os.environ["RWKV_CUDA_ON"] = "1"

    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE, PIPELINE_ARGS

    model_path = hf_hub_download(repo_id=repo_id, filename=f"{title}.pth")
    model = RWKV(model=model_path, strategy=strategy)
    pipeline = PIPELINE(model, tokenize_path)

    def generator(ctx, max_length):
        args = PIPELINE_ARGS(
            temperature=1.0,
            top_p=0.7,
            alpha_frequency=0.1,
            alpha_presence=0.1,
            token_ban=[],
            token_stop=[0],
        )

        gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)

        all_tokens = []
        out_last = 0
        out_str = ""
        occurrence = {}
        state = None
        for i in range(int(max_length)):
            out, state = model.forward(
                pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state
            )
            for n in occurrence:
                out[n] -= args.alpha_presence + occurrence[n] * args.alpha_frequency

            token = pipeline.sample_logits(
                out, temperature=args.temperature, top_p=args.top_p
            )
            if token in args.token_stop:
                break
            all_tokens += [token]
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            tmp = pipeline.decode(all_tokens[out_last:])
            if "\ufffd" not in tmp:
                out_str += tmp
                out_last = i + 1
        gc.collect()
        torch.cuda.empty_cache()
        return [{"generated_text": out_str.strip()}]

    return generator
