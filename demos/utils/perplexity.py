import torch
import os
from tqdm import tqdm

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

# 设置可见 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_dir = "/home/model/Qwen2.5-Coder-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
# 加载模型（自动分配到多卡）
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto"
)

device = Accelerator().device

def perplexity(model, tokenizer, text, target_range=None):
    """
    text: 整个输入序列
    target_range: (start_idx, end_idx)，只计算该区间 token 的 NLL
    """
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)

    # 复制作为标签
    labels = input_ids.clone()

    if target_range is not None:
        start, end = target_range
        # 非目标 token 全部 mask 掉
        mask = torch.ones_like(labels, dtype=torch.bool)
        mask[:, start:end] = False
        labels[mask] = -100

    with torch.no_grad():
        output = model(input_ids, labels=labels)
        loss = output.loss  # 这是平均 NLL

    return loss.item(), torch.exp(loss).item()

def conditional_ppxt(context,target):
    full_text = context + target
    # 找出 target 在 token 中的位置
    enc_context = tokenizer(context, return_tensors="pt")
    enc_full = tokenizer(full_text, return_tensors="pt")
    
    start = enc_context.input_ids.size(1)      # target 开始位置
    end = enc_full.input_ids.size(1)           # target 结束位置

    # 有前文（条件）
    nll_cond, ppl_cond = perplexity(
        model, tokenizer,
        full_text,
        target_range=(start, end)
    )

    # 输出结果
    print("=== Perplexity 对比 ===")
    print(f"有前文条件 PPL: {ppl_cond:.4f}")
    
    return ppl_cond

if __name__=="__main__":
    print()
    