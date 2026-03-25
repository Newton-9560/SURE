import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from safetensors.numpy import save_file
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.utils import api_call, detect_refusal, load_model_and_tokenizer, get_hidden_states, parse_size_list, safety_only_prompt, safety_only_prompt_with_reason
from dataset.main import _MODEL_TEMPLATE_MAP
from dataset.main import SafetyDataset

def inference_openai(model_name, dataset, args):
    '''
    dataset: SafetyDataset.get_dataset()
    '''
    max_new_tokens = args.max_new_tokens
    max_workers = args.max_workers
    print(f"Running inference for {len(dataset)} samples with {max_workers} workers")

    def process_entry(entry):
        """Process a single entry: generate response and detect refusal"""
        result = api_call(model_name, entry["prompt"], max_new_tokens)
        entry["response"] = result["response"]
        entry["refusal"] = detect_refusal(result["response"])
        entry["logprobs"] = result["logprobs_list"]
        return entry

    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_entry = {executor.submit(process_entry, entry): entry for entry in dataset}

        # Process completed tasks with progress bar
        results = []
        for future in tqdm(as_completed(future_to_entry), total=len(dataset), desc=f"Inference ({model_name})"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                entry = future_to_entry[future]
                print(f"Error processing entry {entry.get('id', 'unknown')}: {e}")
                entry["response"] = ""
                entry["refusal"] = True
                results.append(entry)

    # Calculate refusal statistics
    safe_refusal = []
    unsafe_refusal = []
    for entry in results:
        if entry["label"] == "safe":
            safe_refusal.append(entry["refusal"])
        else:
            unsafe_refusal.append(entry["refusal"])

    print(f"Safe refusal: {sum(safe_refusal)} / {len(safe_refusal)}")
    print(f"Unsafe refusal: {sum(unsafe_refusal)} / {len(unsafe_refusal)}")
    return results


def inference_openai_multi(model_name, dataset, num_generations, args):
    '''
    dataset: SafetyDataset.get_dataset()
    '''
    max_new_tokens = 64
    max_workers = args.max_workers
    print(f"Running inference for {len(dataset)} samples with {max_workers} workers")

    def process_entry(entry):
        """Process a single entry: generate response and detect refusal"""
        responses = []
        logprobs = []
        prompt = safety_only_prompt_with_reason.format(prompt=entry["prompt"])
        for i in range(num_generations):
            results = api_call(model_name, prompt, max_new_tokens)
            responses.append(results["response"])
            logprobs.append(results["logprobs_list"])
            # print(results["response"])
            # print(results["logprobs_list"])
            # print("-" * 100)
            # import pdb; pdb.set_trace()
        entry["responses"] = responses
        entry["logprobs"] = logprobs
        entry["refusal"] = True if 'unsafe' in responses[0].lower() else False
        return entry

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_entry = {executor.submit(process_entry, entry): entry for entry in dataset}

        # Process completed tasks with progress bar
        results = []
        for future in tqdm(as_completed(future_to_entry), total=len(dataset), desc=f"Inference ({model_name})"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                entry = future_to_entry[future]
                print(f"Error processing entry {entry.get('id', 'unknown')}: {e}")
                entry["responses"] = []
                entry["logprobs"] = []
                entry["refusal"] = True
                results.append(entry)

    # Calculate refusal statistics
    safe_refusal = []
    unsafe_refusal = []
    for entry in results:
        if entry["label"] == "safe":
            safe_refusal.append(entry["refusal"])
        else:
            unsafe_refusal.append(entry["refusal"])

    print(f"Safe refusal: {sum(safe_refusal)} / {len(safe_refusal)}")
    print(f"Unsafe refusal: {sum(unsafe_refusal)} / {len(unsafe_refusal)}")

    return results

@torch.no_grad()
def inference_local(model_name, dataset, args):
    '''
    dataset: SafetyDataset.get_dataset()
    Uses a local model loaded via load_model_and_tokenizer.
    Processes entries in batches for efficiency on GPU.
    '''
    model, tokenizer = load_model_and_tokenizer(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batch generation

    batch_size = args.batch_size
    max_new_tokens = 10
    results = []
    safety_template = safety_only_prompt

    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Inference local ({model_name})"):
        batch = dataset[i : i + batch_size]
        messages_list = [[{"role": "user", "content": safety_template.format(prompt=entry["prompt"])}] for entry in batch]
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        for j, entry in enumerate(batch):
            new_tokens = outputs[j][input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            # print(response) # DEBUG
            # print("-" * 100) 
            # import ipdb; ipdb.set_trace()
            entry["response"] = response
            if 'unsafe' in response.lower():
                entry["refusal"] = True
            else:
                entry["refusal"] = False
            results.append(entry)

    # Detect refusal for all entries in parallel
    # max_workers = getattr(args, "max_workers", 5)
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     future_to_entry = {executor.submit(detect_refusal, entry["response"]): entry for entry in results}
    #     for future in tqdm(as_completed(future_to_entry), total=len(results), desc="Detecting refusal"):
    #         entry = future_to_entry[future]
    #         try:
    #             entry["refusal"] = future.result()
    #         except Exception as e:
    #             print(f"Error detecting refusal for entry {entry.get('id', 'unknown')}: {e}")
    #             entry["refusal"] = True

    safe_refusal = []
    unsafe_refusal = []
    for entry in results:
        if entry["label"] == "safe":
            safe_refusal.append(entry["refusal"])
        else:
            unsafe_refusal.append(entry["refusal"])

    print(f"Safe refusal: {sum(safe_refusal)} / {len(safe_refusal)}")
    print(f"Unsafe refusal: {sum(unsafe_refusal)} / {len(unsafe_refusal)}")
    return results


def save_hidden_states(model_name, dataset):
    model, tokenizer = load_model_and_tokenizer(model_name)
    save_dir = f"outputs/hidden_states/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    for entry in tqdm(dataset, desc=f"Hidden states ({model_name})"):
        messages = [{"role": "user", "content": entry["prompt"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        hidden_states = get_hidden_states(model, tokenizer, prompt)
        entry_id = entry["id"]
        dataset_name = entry["dataset_name"]
        filename = f"{dataset_name}_{entry_id}.safetensors"
        save_file({"hidden_states": hidden_states}, os.path.join(save_dir, filename))
    print(f"Hidden states saved to {save_dir}/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama3", choices=list(_MODEL_TEMPLATE_MAP.keys()))
    parser.add_argument("--dataset", nargs='+', default=["figtxt", "wildjailbreak_vanilla", "wildjailbreak_adversarial", "jbb_behaviors"])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--module", "-m", type=str, default='generate response', choices=['generate response', 'generate response local', 'save hidden states'])
    parser.add_argument("--labeled_size", "-ls", nargs="+", default=[0.5, 0.5, 0.5, 0.5],
                        help="Labeled samples per dataset (int) or ratio (float < 1)")
    parser.add_argument("--val_size", "-vs", nargs="+", default=[0.5, 0.5, 0.5, 0.5],
                        help="Val samples per dataset (int) or ratio (float < 1)")
    parser.add_argument("--max_workers", type=int, default=5, help="Number of parallel workers for API calls")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for local model inference")
    parser.add_argument("--num_generations", "-ng", type=int, default=1, help="Number of generations for each model")
    args = parser.parse_args()

    labeled_size = parse_size_list(args.labeled_size)
    val_size     = parse_size_list(args.val_size)

    if args.module == 'generate response':
        dataset = SafetyDataset(labeled_size=labeled_size, val_size=val_size, dataset_name_list=args.dataset)
        data = dataset.get_dataset(dataset_name_list=args.dataset)
        if args.num_generations == 1:
            data = inference_openai(args.model_name, data, args)
        else:
            data = inference_openai_multi(args.model_name, data, args.num_generations, args)

        os.makedirs("outputs", exist_ok=True)
        save_path = f"outputs/{args.model_name}_responses_{len(data)}.json"
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {save_path}")
    elif args.module == 'generate response local':
        dataset = SafetyDataset(labeled_size=labeled_size, val_size=val_size, dataset_name_list=args.dataset)
        data = dataset.get_dataset(dataset_name_list=args.dataset)
        data = inference_local(args.model_name, data, args)

        os.makedirs("outputs", exist_ok=True)
        save_path = f"outputs/{args.model_name}_responses_local_{len(data)}.json"
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {save_path}")
    elif args.module == 'save hidden states':
        dataset = SafetyDataset(labeled_size=labeled_size, val_size=val_size, dataset_name_list=args.dataset)
        data = dataset.get_dataset(dataset_name_list=args.dataset)
        save_hidden_states(args.model_name, data)

if __name__ == "__main__":
    main()