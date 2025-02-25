from datasets import load_dataset
import vllm
import torch
from transformers import AutoTokenizer
import json


def generate_prompt(tokenizer, question):
    messages = [
        #{ "role": "system", "content": system_prompt},
        { "role": "user", "content": question },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_rephrase_prompt(tokenizer, question, answer):
    messages = [
        #{ "role": "system", "content": "Your job is to rephrase the final answer in a presented question/answer pair as a different but equivalent expression or phrase. This can include adding, changing, or removing words, and also utilizing equivalent but different mathematical expressions (fractions to decimals, simplifying or expanding, separating out variables, etc.). Only provide the final answer rephrased. You will provide three total different rephrasings."},
        { "role": "system", "content": "Your job is to rephrase the final answer in a presented question/answer pair as a different but equivalent expression or phrase. This can include adding, changing, or removing words, and also utilizing equivalent but different mathematical expressions (fractions to decimals, simplifying or expanding, separating out variables, etc.). Only provide the final answer rephrased. Be very creative and make the rephrases all very different looking. They can also greatly vary in length and complexity. You will provide three total different rephrasings."},
        { "role": "user", "content": f"Original Question:\n\n{question}\n\nOriginal Answer:\n\n{answer}\n" },
        { "role": "assistant", "content": "I will now provide 3 rephrasings of the final answer.\n\nRephrased Final Answer 1:\n\n"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)


model_name = "Qwen/Qwen2.5-32B-Instruct"
model = vllm.LLM(model_name, dtype=torch.bfloat16, tensor_parallel_size=2, device="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

dataset = load_dataset("json", data_files="/new_data/data_vault/reasoning/limo_cleaned.jsonl", split="train")
dataset = dataset.shuffle(seed=42).select(range(30))


sampling_params = vllm.SamplingParams(
    max_tokens=8192,
    temperature=0,
)


prompts = []
for datum in dataset:
    prompts.append(generate_prompt(tokenizer, datum["problem"]))

samples = []
responses = model.generate(prompts,sampling_params)
for i, response in enumerate(responses):
    question = dataset[i]["problem"]
    gt = dataset[i]["answer"]
    gt_answer_only = gt.split("**Final Answer**")[1].lstrip() #will only work with LIMO and other similarly formatted data
    generated = response.outputs[0].text.strip()

    rephrase_response = model.generate(generate_rephrase_prompt(tokenizer, question, gt), sampling_params=sampling_params)[0].outputs[0].text
    print("-"*20)
    print(gt_answer_only)
    rephrases = rephrase_response.split("\n\n")
    rephrases = [rephrase for i, rephrase in enumerate(rephrases) if i % 2 == 0]
    assert len(rephrases) == 3

    samples.append({"problem": question, "generated": generated, "gt": gt})
    samples.append({"problem": question, "generated": generated, "gt": gt_answer_only})
    for rephrase in rephrases:
       samples.append({"problem": question, "generated": generated, "gt": rephrase})
       print("-------")
       print(rephrase)

with open("gen_and_gt.jsonl", 'w') as data_file:
    data_file.write('\n'.join(map(json.dumps, samples)))


#longform
#final answer
#rephrase
