import json
import torch
import numpy as np
options = ["A", "B", "C", "D"]



def read_jsonl(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        for item in file:
            if item.strip():
                data.append(json.loads(item))
    return data

def format_example(question, choices, answer, include_answer=True):
    
    prompt = question
    # print(len(choices))
    # print(choices)
    for j in range(len(choices)):
        prompt += "\n{}. {}".format(options[j], choices[j])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(options[answer])
    return prompt

def gen_prompt(k=5):
    prompt = "The following are multiple choice questions.\n\n"

    few_shot_data = read_jsonl('../data/mmlu_shot.jsonl')
    few_shot_data = few_shot_data[:k]
    if k != 0:
        prompt += 'Here are some examples.\n'
    for i in range(min(k,len(few_shot_data))):
        prompt += format_example(few_shot_data[i]['question'], few_shot_data[i]['choices'], few_shot_data[i]['answer'])
    return prompt

@torch.no_grad()
def eval(model, tokenizer, eval_datas):
    questions = [i['mmlu_questions'] for i in eval_datas]
    choices = [i['mmlu_choices'] for i in eval_datas]
    answer = [i['mmlu_answer'] for i in eval_datas]
    # print(questions)
    cors = []
    all_probs = []
    all_answers = []

    for i in range(len(questions)): #batch_size
        temp_cors = []
        temp_probs = []
        temp_answers = []
        for j in range(len(questions[i])):
        # get prompt and make sure it fits
        # k = args.ntrain
            prompt_end = format_example(questions[i][j], choices[i][j], answer[i][j], include_answer=False)
            train_prompt = gen_prompt(5)
            prompt = train_prompt + prompt_end

            # print(prompt)

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            # model.eval()

            logits = model(
                input_ids=input_ids
            ).logits[0,-1] #bs:seq:vocab_size


            # probs = (
            #     # torch.nn.functional.softmax(
            #         torch.tensor(
            #             [
            #                 logits[tokenizer("A").input_ids[0]],
            #                 logits[tokenizer("B").input_ids[0]],
            #                 logits[tokenizer("C").input_ids[0]],
            #                 logits[tokenizer("D").input_ids[0]],
            #             ]
            #         )
            #     .detach()
            #     .cpu()
            #     .numpy()
            # )
            
            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer("A").input_ids[-1]],
                            logits[tokenizer("B").input_ids[-1]],
                            logits[tokenizer("C").input_ids[-1]],
                            logits[tokenizer("D").input_ids[-1]],
                        ]
                    ),
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

            cor = np.argmax(probs) == answer[i]
            temp_cors.append(cor)
            temp_probs.append(probs)
            temp_answers.append(pred)
        cors.append(temp_cors)
        all_probs.append(temp_probs)
        all_answers.append(temp_answers)


    # acc = np.mean(cors)
    # cors = np.array(cors)

    # all_probs = np.array(all_probs)
    # print("Average accuracy {:.3f}".format(acc))

    return cors, all_probs, all_answers