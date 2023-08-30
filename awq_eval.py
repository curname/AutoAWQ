import time
import re
import json
import torch
import argparse
import numpy as np
from loguru import logger
from tqdm import tqdm
from awq.models import *
from awq.models.auto import AutoAWQForCausalLM
from attributedict.collections import AttributeDict
# from tinychat.utils.prompt_templates import get_prompter, get_stop_token_ids
# from tinychat.stream_generators import StreamGenerator, FalconStreamGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, modeling_utils, GenerationConfig

# opt_params in TinyLLMEngine
gen_params = AttributeDict([
                    ("seed", -1),               # RNG seed
                    ("n_threads", 1),           # TODO: fix this
                    ("n_predict", 512),         # new tokens to predict
                    ("n_parts", -1),            # amount of model parts (-1: determine from model dimensions)
                    ("n_ctx", 1024),             # context size
                    ("n_batch", 512),           # batch size for prompt processing (must be >=32 to use BLAS)
                    ("n_keep", 0),              # number of tokens to keep from initial prompt
                    ("n_vocab", 50272),         # vocabulary size

                    # sampling parameters
                    ("logit_bias", dict()),     # logit bias for specific tokens: <int, float>
                    ("top_k", 40),              # <= 0 to use vocab size
                    ("top_p", 0.95),            # 1.0 = disabled
                    ("tfs_z", 1.00),            # 1.0 = disabled
                    ("typical_p", 1.00),        # 1.0 = disabled
                    ("temp", 0.20),             # 1.0 = disabled
                    ("repeat_penalty", 1.00),   # 1.0 = disabled
                    ("repeat_last_n", 64),      # last n tokens to penalize (0 = disable penalty, -1 = context size)
                    ("frequency_penalty", 0.00),# 0.0 = disabled
                    ("presence_penalty", 0.00), # 0.0 = disabled
                    ("mirostat", 0),            # 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
                    ("mirostat_tau", 5.00),     # target entropy
                    ("mirostat_eta", 0.10),     # learning rate
                ])

def stream_output(output_stream, total_time):
    print(f"ASSISTANT: ", end="", flush=True)
    pre = 0
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            # print(" ".join(output_text[pre:now]), end=" ", flush=True)
            pre = now
    # print(" ".join(output_text[pre:]), flush=True)
    if "timing" in outputs and outputs["timing"] is not None:
        timing = outputs["timing"]
        context_tokens = timing["context_tokens"]
        context_time = timing["context_time"]
        total_tokens = timing["total_tokens"]
        generation_time_list = timing["generation_time_list"]
        generation_tokens = len(generation_time_list)
        average_speed = (context_time + np.sum(generation_time_list)) / (context_tokens + generation_tokens)
        print("=" * 50)
        print("Speed of Inference")
        print("-" * 50)
        # print(f"Context Stage    : {context_time/context_tokens * 1000:.2f} ms/token")
        print(f"Generation Stage : {np.average(generation_time_list) * 1000:.2f} ms/token")
        # print(f"Average Speed    : {average_speed * 1000:.2f} ms/token")
        print("=" * 50)
        # print("token num:", total_tokens)
        # print("Model total Time = ", (context_time + np.sum(generation_time_list))*1000, "ms" )
        total_time.append(np.average(generation_time_list) * 1000)
    return outputs["text"]
    # return " ".join(output_text)

def device_warmup(device:str):
    warm_up = torch.randn((4096,4096)).to(device)
    torch.mm(warm_up,warm_up)

def process_brackets(raw_code: str, prompt: str) -> str:
    code = raw_code[len(prompt):]
    stack = ['{']
    for index, char in enumerate(code):
        if char == '{':
            stack.append(char)
        elif char == '}':
            if not stack:
                return False
            stack.pop()
        if not stack:
            return code[:index] + "}"
    return code

# def process_python(raw_code: str, prompt: str) -> str:
#     # logger.info("\n" + raw_code)
#     filename_prefix = "<filename>"
#     if not raw_code.startswith(prompt) and prompt.startswith(filename_prefix):
#         prompt = prompt[len(filename_prefix):]
#     code = raw_code[len(prompt):]
#     code_lines = code.split("\n")
#     new_lines = []
#     doc_str_block = False
#     doc_str_bracket = "\"\"\""
#     for line in code_lines:
#         if line == "​":
#             continue
#         if not doc_str_block:
#             if len(line) > 0 and len(line) == len(line.lstrip()):
#                 if re.match("[^\s\S]+", line):
#                     continue
#                 if re.match("[\"]{3}", line):
#                     if not re.match("[\"]{3}.*[\"]{3}", line):
#                         doc_str_block = True
#                 elif re.match("[\']{3}", line):
#                     if not re.match("[\']{3}.*[\']{3}", line):
#                         doc_str_block = True
#                         doc_str_bracket = "\'\'\'"
#                 elif re.match(".*", line) and not line.startswith("#") and not line.startswith("def "):
#                     break
#         else:
#             if line.endswith(doc_str_bracket):
#                 doc_str_block = False
#         new_lines.append(line)
#     ret = "\n".join(new_lines)
#     return ret

def process_python(raw_code: str, prompt: str, stream: bool) -> str:
    if stream:
        code = raw_code[:]
    else:
        code = raw_code[len(prompt):]
    code_lines = code.split("\n")
    new_lines = []
    for line in code_lines:
        if len(line) > 0 and len(line) == len(line.lstrip(" ")):
            break
        new_lines.append(line)
    return "\n".join(new_lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data/llm/checkpoints/vicuna-hf/vicuna-7b', help='path to the model')
    parser.add_argument('--quant_file', type=str, default='awq_model_w4_g128.pt', help='path to the model file')
    parser.add_argument('--precision' , type=str, default='W4A16', help='compute precision')
    parser.add_argument('--device'    , type=str, default='cuda')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--stream', action="store_true", default="")
    parser.add_argument('--model_type', type=str, default="pangucoder2", help="")

    args = parser.parse_args()
    assert args.precision in ["W4A16", "W16A16"], "We only support W4A16/W16A16 now"

    gen_params.n_predict = 512
    gen_params.n_vocab = 32000

    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if "mpt" in config.__class__.__name__.lower():
        # config.init_device="meta"
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    if args.precision == "W4A16":
        model = AutoAWQForCausalLM.from_quantized(args.model_path, args.quant_file, device={"":args.device}, fuse_layers=False)
        assert model.model_type.lower() in ["llama", "refinedweb", "refinedwebmodel", "mpt", "gpt_bigcode"], "We only support llama & falcon & mpt now"
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True).to(args.device)

    # device warm up
    device_warmup(args.device)

    if not args.stream:
        generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=False,
            temperature=0.2,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )

        if args.model_type.split("_")[0] == "pangucoder2":
            with open(f'humaneval_x_input/human_eval_all_sft_data-0619-v6.json', 'r') as fr:
                lines = fr.readlines()
                lines = eval(lines[0])["data"]
        else:
            with open(f'humaneval_x_input/he_{args.language}_function_generation_dataset.jsonl', 'r') as fr:
                lines = fr.readlines()
        logger.info(len(lines))

        all_time = 0
        fw = open(f'humaneval_x_output/output_{args.language}_{args.model_type}.jsonl', 'a', encoding='utf-8')
        for i, line in tqdm(enumerate(lines)):
            if args.model_type.split("_")[0] == "pangucoder2":
                task = line
            else:
                task = json.loads(line)

            prompt = task["prompt"]
            x = tokenizer.encode(prompt, return_tensors='pt').to(device=args.device)
            start_time = time.time()
            y = model.generate(inputs=x, generation_config=generation_config)
            end_time = time.time()
            inference_time = end_time - start_time
            per_token_time = inference_time / (y.shape[1] - x.shape[1])
            all_time += per_token_time
            logger.info("###### pre token time :{:.2}ms ######".format(per_token_time * 1000))
            generated_snippet = tokenizer.decode(
                y[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).replace("\u200b", "")
            if args.language in ['java']:
                function_code = process_brackets(generated_snippet, prompt) + "\n}"
            elif args.language in ['go', 'c', 'cpp', 'javascript', 'typescript']:
                function_code = process_brackets(generated_snippet, prompt)
            elif args.language == 'python':
                function_code = process_python(generated_snippet, prompt, args.stream)
            logger.info('\n' + function_code)

            if args.model_type.split("_")[0] == "pangucoder2":
                result = {
                    'task_id': task['question_slug'],
                    'completion': function_code
                }
            else:
                result = {
                    'task_id': task['question_id'],
                    'completion': function_code
                }

            fw.write(json.dumps(result) + '\n')
            fw.flush()
        logger.info("------- pre token time with:{:.2}ms -------".format(all_time * 1000 / 164))
