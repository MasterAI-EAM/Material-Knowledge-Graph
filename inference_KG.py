import time
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import sys
import json
from tqdm import tqdm
import signal
import multiprocessing
import argparse
import os
from peft import PeftModel

# clear cache
torch.cuda.empty_cache()
num_gpus = torch.cuda.device_count()
print(num_gpus)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

############################## data part ##########################################
IO_chunk_size = 50

def load_txt_files(file_path):
    data_items = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line.strip())
            data_items.append((json_data['resource'], json_data['instruction']))
    return data_items


def save_and_exit(signum, frame):
    print('terminating by signal')
    results = list(shared_results)
    with open(filename, "w") as file:
        for i in tqdm(range(0, len(results), IO_chunk_size)):
            chunk = results[i:i + IO_chunk_size]
            file.writelines([json.dumps(item) + "\n" for item in chunk])
        file.flush()
    sys.exit(0)  # Exit gracefully


# Set the signal handler
signal.signal(signal.SIGTERM, save_and_exit)
# Set the signal handler for SIGINT
signal.signal(signal.SIGINT, save_and_exit)

########################## model part ######################################
# inference
def inference(sent, tokenizer, model):
    input_ids = tokenizer(sent, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(
        input_ids=input_ids,
        # max_length=2048,
        max_new_tokens=1024,
        do_sample=True,
        repetition_penalty=1.0,
        temperature=0.8,
        top_p=0.75,
        top_k=40
    )
    res = tokenizer.decode(generated_ids[0])

    del input_ids
    del generated_ids
    return res


def process_on_gpu(gpu_id, data_subset, shared_results, model_path, basemodel_path, use_PEFT_model, counts, output_path):
    try:
        torch.cuda.set_device(gpu_id)

        s = time.time()
        tokenizer = LlamaTokenizer.from_pretrained(model_path, maxlength=2048)
        if not use_PEFT_model:
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=False,
                torch_dtype=torch.float16
            ).cuda(gpu_id)
        else:
            base_model = LlamaForCausalLM.from_pretrained(basemodel_path, load_in_8bit=False,
                 torch_dtype=torch.float16, use_auth_token='your access token')
            peft_model = PeftModel.from_pretrained(base_model, model_path, load_in_8bit=False,
                 torch_dtype=torch.float16,)
            model = peft_model.cuda(gpu_id)

        e = time.time()
        print(f'\nModel load in time on {gpu_id}: {e - s}\n')

        s = time.time()
        count = 0
        results_in_file = []
        for key, data in data_subset:
            res = inference(data, tokenizer, model)
            shared_results.append({'paper': key, 'instruction': data, 'response': res})
            results_in_file.append({'paper': key, 'instruction': data, 'response': res})
            del res
            torch.cuda.empty_cache()

            count += 1
            if count % 10 == 0:
                with open(f'{output_path}/inference{gpu_id}.json', "a") as file:
                    file.writelines([json.dumps(item) + "\n" for item in results_in_file])
                results_in_file = []
                e = time.time()
                print(f'\ngpu {gpu_id}: {count} files has already been inferenced, the inference time is {e - s}\n')
                s = time.time()

            counts.value += 1

    except Exception as e:
        print(f"Error on GPU {gpu_id}: {e}")

    return 0


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process paths.')
    parser.add_argument('--data_path', type=str, help='Path to the data file')
    parser.add_argument('--output_path', type=str, help='Path to save the output')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--timeout', type=int, default=20000, help='the forceable time to stop program')
    parser.add_argument('--basemodel_path', type=str, default='none', help='basemodel for PEFT model')
    parser.add_argument('--use_PEFT', type=bool, default=False, help='if uses PEFT model. If True, basemodel_path must be added')

    # Parse arguments
    args = parser.parse_args()

    multiprocessing.set_start_method('spawn', force=True)
    
    data_path = args.data_path
    output_path = args.output_path
    model_path = args.model_path
    filename = os.path.join(output_path, 'inference.json')
    timeout = args.timeout #seconds
    basemodel_path = args.basemodel_path
    use_PEFT_model = args.use_PEFT

    print("loading model, path:", model_path)

    # Check if output_path exists, if not create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(num_gpus):
        with open(f'{output_path}/inference{i}.json', "w") as file:
            pass

    # Load data from txt files
    data_items = load_txt_files(data_path)
    # data_items = list(data_dic.items())
    num_data = len(data_items)
    chunk_size = len(data_items) // num_gpus + 1
    data_chunks = [data_items[i:i + chunk_size] for i in range(0, len(data_items), chunk_size)]
    print(f'{num_data} data, {num_gpus} gpus, {chunk_size} chunks, chunk len = {len(data_chunks)}')


    # Create a shared list using multiprocessing.Manager
    manager = multiprocessing.Manager()
    shared_results = manager.list()
    counts = manager.Value('i', 0)

    # Use multiprocessing to process each chunk on a separate GPU with a timeout
    pool = multiprocessing.Pool(processes=num_gpus)
    for args in enumerate(data_chunks):
        pool.apply_async(process_on_gpu, args=(args[0], args[1], shared_results, model_path, basemodel_path, use_PEFT_model, counts, output_path))
    pool.close()

    start_time = time.time()
    while True:
        if counts.value >= num_data:
            print('All processes success')
            pool.terminate()
            break
        if time.time() - start_time > timeout:
            print('terminal because of timeout')
            pool.terminate()
            break
        time.sleep(60)

    results = list(shared_results)

    # Flatten the results and write to file
    print('\nWriting into file!!!\n')
    with open(filename, "w") as file:
        try:
            for i in tqdm(range(0, len(results), IO_chunk_size)):
                chunk = results[i:i + IO_chunk_size]
                file.writelines([json.dumps(item) + "\n" for item in chunk])
            file.flush()
        except Exception as e:
            file.flush()
            print(f"An error occurred: {e}")

