import json
import argparse
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser(description="Certain/Uncertain Classifier")
    parser.add_argument("--dataset_name", type=str, default="HotpotQA", help="HotpotQA, StrategyQA, MATH500, T4D")
    parser.add_argument("--data_dir", type=str, default="data")

    args = parser.parse_args()

    data_dir = os.path.join(args.data_dir, args.dataset_name)
    output_dir = os.path.join(args.data_dir, args.dataset_name)
    output_path = os.path.join(output_dir, f'{args.dataset_name}.json')
    os.makedirs(output_dir, exist_ok=True)

    c_dataset = []

    if args.dataset_name == "HotpotQA":
        data_path = os.path.join(data_dir, "hotpot_dev_distractor_v1.json")
        if not os.path.exists(data_path):
            raise ValueError(f"Dataset not found: {os.path.abspath(data_path)} (expected in {data_dir})")
        with open(data_path, 'r') as f:
            dataset = json.load(f)
        
        for idx, data in enumerate(dataset):
            tmp = {'id': idx,
                   'question': data['question'],
                   'answer': data['answer'],
                   'context': data['context']}
            c_dataset.append(tmp)
    
    elif args.dataset_name == "StrategyQA":
        data_path = os.path.join(data_dir, "dev.json")
        if not os.path.exists(data_path):
            raise ValueError(f"Dataset not found: {os.path.abspath(data_path)} (expected in {data_dir})")
        with open(data_path, 'r') as f:
            dataset = json.load(f)
        
        for idx, data in enumerate(dataset):
            tmp = {'id': idx,
                   'question': data['question'],
                   'answer': data['answer'],
                   'facts': data['facts'],
                   'decomposition': data['decomposition'],
                   'evidence': data['evidence']}
            c_dataset.append(tmp)

    elif args.dataset_name == "Musique":
        data_path = os.path.join(data_dir, "musique_full_v1.0_dev.jsonl")
        if not os.path.exists(data_path):
            raise ValueError(f"Dataset not found: {os.path.abspath(data_path)} (expected in {data_dir})")
        with open(data_path, 'r') as f:
            dataset = []
            for data in f:
                dataset.append(json.loads(data))
        
        for idx, data in enumerate(dataset):
            tmp = {'id': idx,
                   'question': data['quesiton'],
                   'answer': data['answer'],
                   'answerable': data['answerable'],
                   }

    elif args.dataset_name == "T4D":
        data_path = os.path.join(data_dir, "train-00000-of-00001.parquet")
        if not os.path.exists(data_path):
            raise ValueError(f"Dataset not found: {os.path.abspath(data_path)} (expected in {data_dir})")
        df = pd.read_parquet(data_path)

        dataset = df.to_json(orient='records')
        dataset = json.loads(dataset)

        for idx, data in enumerate(dataset):
            tmp = {'id': idx}
            for k, v in data.items():
                tmp[k] = v
            tmp['question'] = " ".join([data['story'], data['question']])
            del tmp['story']
            c_dataset.append(tmp)

    elif args.dataset_name == "MATH500":
        data_path = os.path.join(data_dir, 'test.jsonl')
        if not os.path.exists(data_path):
            raise ValueError(f"Dataset not found: {os.path.abspath(data_path)} (expected in {data_dir})")
        with open(data_path, 'r') as f:
            for idx, line in enumerate(f):
                tmp = json.loads(line)
                c_dataset.append({'id': idx,
                                  'question': tmp['problem'],
                                  'answer': tmp['answer']})
        print("")
    
    else:
        raise ValueError(f"Dataset not found: {os.path.abspath(data_path)} (expected in {data_dir})")

    with open(output_path, 'w') as f:
        json.dump(c_dataset)
    print(f"{args.dataset_name} successfully converted and saved.")
                

if __name__ == "__main__":
    main()
