import json
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import argparse
import sys
sys.setrecursionlimit(2000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default='../output/result_seq64_bs1_ex20_step50_s_20.json', type=str)
    parser.add_argument('--model_path', default='../model/all-MiniLM-L6-v2', type=str)
    parser.add_argument('--device', default=0, type=int)

    args = parser.parse_args()
    with open(args.file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    # for i in data:
    #     if i['original_prediction'] == '':
    #         i['original_prediction'] = ' '
    #     if i['para_prediction'] == '':
    #         i['para_prediction'] = ' '
    for i in data:
        if ' ' not in i['original_prediction']:
            i['original_prediction'] += ' '
        if ' ' not in i['para_prediction']:
            i['para_prediction'] += ' '
    matrics_all = []
    for num in range(1,len(data)+1):

        
        matrics = {}

        # cal bleu
        bleu_scores = []
        rouge1s=[]
        rouge2s=[]
        rougels=[]
        
        bleu_scores_para = []
        rouge1s_para=[]
        rouge2s_para=[]
        rougels_para=[]
        rouge = Rouge()

        for index in tqdm(range(len(data[:num])),desc='Calculate BLEU&ROUGE Score'):
            score = sentence_bleu([data[index]['answer']], data[index]['original_prediction'])
            bleu_scores.append(score)

            score = sentence_bleu([data[index]['answer']], data[index]['para_prediction'])
            bleu_scores_para.append(score)

            scores = rouge.get_scores(data[index]['original_prediction'],data[index]['answer'])
            rouge1s.append(scores[0]['rouge-1']['r'])
            rouge2s.append(scores[0]['rouge-2']['r'])
            rougels.append(scores[0]['rouge-l']['r'])

            scores = rouge.get_scores(data[index]['para_prediction'],data[index]['answer'])
            rouge1s_para.append(scores[0]['rouge-1']['r'])
            rouge2s_para.append(scores[0]['rouge-2']['r'])
            rougels_para.append(scores[0]['rouge-l']['r'])
        temp_original = {}
        temp_para = {}
        temp_original['BLEU SCORE'] = sum(bleu_scores) / len(bleu_scores)
        temp_original['ROUGE-1'] = sum(rouge1s) / len(rouge1s)
        temp_original['ROUGE-2'] = sum(rouge2s) / len(rouge2s)
        temp_original['ROUGE-L'] = sum(rougels) / len(rougels)

        temp_para['BLEU SCORE'] = sum(bleu_scores_para) / len(bleu_scores_para)
        temp_para['ROUGE-1'] = sum(rouge1s_para) / len(rouge1s_para)
        temp_para['ROUGE-2'] = sum(rouge2s_para) / len(rouge2s_para)
        temp_para['ROUGE-L'] = sum(rougels_para) / len(rougels_para)
        # cal bert score
        print("***********Calculate BERT Similarity Score**************")
        sentences1 = [i['answer'] for i in data[:num]]
        sentences2 = [i['original_prediction'] for i in data[:num]]
        sentences3 = [i['para_prediction'] for i in data[:num]]
        model = SentenceTransformer(args.model_path, device=f"cuda:{args.device}")

        embeddings1 = model.encode(sentences1, convert_to_tensor=True,show_progress_bar=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True,show_progress_bar=True)
        embeddings3 = model.encode(sentences3, convert_to_tensor=True,show_progress_bar=True)
        # if index == 1:
        # print(index)
        # print(sentences1)
        # print(embeddings1)
        # Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        # print(cosine_scores)
        # print(cosine_scores.shape)
        temp_original['Bert Score'] = cosine_scores.diagonal().mean().item()

        cosine_scores = util.cos_sim(embeddings1, embeddings3)
        # print(cosine_scores.shape)
        temp_para['Bert Score'] = cosine_scores.diagonal().mean().item()
        
        matrics['Original']=temp_original
        matrics['Para']=temp_para

        matrics_all.append(matrics)
        # break
        # print("***********Result**************")
        # print(matrics)
    with open("result_reshape_unke_64_seq.json", "w", encoding="utf-8") as file:
        json.dump(matrics_all,file,ensure_ascii=False,indent=4)

