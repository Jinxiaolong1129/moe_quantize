import sys
sys.path.append("/home/LeiFeng/xiaolong/moe_quantize/optimum/")  # Add the path to Python's search path
print(sys.path)


from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model, GPTQQuantizer_deepseek
import torch
import random
from argparse import ArgumentParser

from transformers import AutoTokenizer, TextGenerationPipeline
import logging
from datasets import load_dataset

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig, AutoGPTQForCausalLM_mixed_precision, BaseQuantizeConfig_mixed_precision
import logging


def get_wikitext2(tokenizer, seqlen: int, nsamples: int, split: str = "train"):
    if split == "train":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif split == "validation":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    text = "".join([" \n" if s == "" else s for s in data["text"][:1000]])

    enc = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return dataset


def get_Pile_dataset(tokenizer, seqlen: int, nsamples: int, split: str = "train"):
    data = load_dataset("json", data_files='data/minipile/val.jsonl.zst', split="train")


    text = "".join([" \n" if s == "" else s for s in data["text"][:1000]])

    enc = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return dataset


def main():
    parser = ArgumentParser()
    parser.add_argument("--bits", type=int, default="4")
    
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--quant_path", type=str, default=None)
    
    parser.add_argument("--nsamples", type=int, default=512)
    parser.add_argument("--seqlen", type=int, default=512)  

    parser.add_argument("--group_size", type=int, default=128)    
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f"/home/LeiFeng/xiaolong/moe_quantize/quantize_gptq_deepseek_{args.bits}.log"
    )
        

    args_dict = vars(args)
    logging.info("Command-line arguments: %s", args_dict)

    model_name = args.model_name    
    quant_path = f'autogptq_{model_name}-gptq_w_bit_{args.bits}'
    
    quantize_config = BaseQuantizeConfig(
        bits=args.bits,  # quantize model to 4-bit
        group_size=args.group_size,  # default 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_dataset = get_Pile_dataset(tokenizer=tokenizer, seqlen=args.seqlen, nsamples=args.nsamples, split="train")

    model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config, 
                                                torch_dtype=torch.float16, trust_remote_code=True)
    
    logging.info(f"Quantization dataset loaded with {args.nsamples} samples")
    logging.info(f"Quantizing model to {args.bits}-bit")
    logging.info(f"Quantized begin!!!!")
    model.quantize(quantization_dataset)
    logging.info(f"Quantized finish!!!!")
    
    logging.info(f"Quantized model begin to save")
    model.save_quantized(quant_path)
    logging.info(f"Quantized model saved to {quant_path}")



if __name__ == "__main__":

    main()