import os
import sys
import subprocess

# parameters
DEVICE_TYPE = 'gpu' # cpu or gpu
GPU = '0'
BASICTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
BASELINE_PREFIX = 'baselines/TimeMoE/evaluate_config/'

DATASET_LIST = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']
MODEL_LIST = ['base.py', 'large.py']
CHECKPOINT_PATH_List = ['/path/to/your/base/checkpoint.pt','/path/to/your/large/checkpoint.pt']
CONTEXT_LENGTH_LIST = [512, 720, 1024, 2048]
PREDICTION_LENGTH_LIST = [96, 192, 336, 720]


def gen_config_path(dataset, model):
    config_path = os.path.join(BASELINE_PREFIX, dataset, model)
    return config_path

def gen_evaluate_command(checkpoint, config_path, context_length, prediction_length):
    command = f'cd {BASICTS_ROOT}; python experiments/evaluate.py -ckpt "{checkpoint}" -d {DEVICE_TYPE} -cfg "{config_path}" -g {GPU} -ctx {context_length} -pred {prediction_length}'
    return command

if __name__ == '__main__':
    eval_result = {}
    logs = []

    for dataset in DATASET_LIST:
        for model, checkpoint in zip(MODEL_LIST, CHECKPOINT_PATH_List):
            config_path = gen_config_path(dataset, model)
            for context_length in CONTEXT_LENGTH_LIST:
                for prediction_length in PREDICTION_LENGTH_LIST:
                    command = gen_evaluate_command(checkpoint, config_path, context_length, prediction_length)
                    print(command)
                    # get output
                    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
                    ouput = result.stderr
                    logs.append(ouput)
                    print(ouput)

                    # find number in "Result <test>: [test/time: 9.70 (s), test/MAE: 0.3774, test/MSE: 0.3671]"
                    for line in ouput.split('\n'):
                        if 'Result <test>:' in line:
                            metrics = line.split('[')[1].split(']')[0]
                            metric_list = metrics.split(',')
                            mae = float(metric_list[1].split(':')[1].strip())
                            mse = float(metric_list[2].split(':')[1].strip())
                            eval_result[f'{dataset}_{model}_ctx{context_length}_pred{prediction_length}'] = {'MAE': mae, 'MSE': mse}
                            break

    # write logs to file
    with open('timemoe_evaluation_logs.txt', 'w') as f:
        for log in logs:
            f.write(log)
            f.write('\n' + '='*80 + '\n')

    # write eval_result to file
    with open('timemoe_evaluation_results.txt', 'w') as f:
        for key, value in eval_result.items():
            f.write(f'{key}: MAE={value["MAE"]}, MSE={value["MSE"]}\n')
    
    with open('timemoe_evaluation_results.json', 'w') as f:
        import json
        json.dump(eval_result, f, indent=4)
