import openai
import time
import concurrent.futures
import numpy as np
from tqdm import tqdm
import json
def single_run(que, ans, p, retry=5, model="gpt-4o", n=1, temperature=0.):
    for _ in range(retry):
        try:
            output = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are given a question, a correct answer, and a prediction.  You need to judge whether the prediction fits the correct answer. You just need to answer yes or no."},
                    {"role": "user", "content": f"Question:{que}\n Correct Answer:{ans}\n Prediction:{p} "},
                ],
                n=n,
                temperature=temperature,
            )

            if n == 1:
                return output.choices[0].message.content.strip()
            else:
                return [choice.message.content for choice in output.choices]
        except:
            time.sleep(20)
    return None

def fact_score_average(que, ans, p, runs=5):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda _: single_run(que, ans, p), range(runs)))
    
    
    valid_results = [1 if res.lower() == "yes" else 0 for res in results if res is not None]
    
    if valid_results:
        return np.mean(valid_results)
    return None

def process_data(data):
    results = []
    for i in tqdm(range(len(data))):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = list(executor.map(
                lambda j: fact_score_average(
                    data[i]['sub_question'][j], 
                    data[i]['sub_answer'][j], 
                    data[i]['sub_pred'][j]
                ), 
                range(len(data[i]['sub_question']))
            ))
        results.append(result)
    return results

def replace_none_with_no(lst):
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            if lst[i][j] is None:
                lst[i][j] = 'No'

def evaluate_results(results):
    replace_none_with_no(results)
    result_sub_score = []
    result_sub_len = []
    for i in results:
        result_score = [1 if 'Yes' in j or 'yes' in j else 0 for j in i]
        result_sub_score.append(sum(result_score)/len(result_score))
        result_sub_len.append(len(result_score))
    return sum(result_sub_score)/len(result_sub_score)

with open('your_data_path.json', 'r') as file:
    data = json.load(file)
final_results = process_data(data)
final_score = evaluate_results(final_results)
print(final_score)
