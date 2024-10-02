# UnKE
This is the official repo for our [paper](https://arxiv.org/abs/2405.15349): 
> Everything is Editable: Extend Knowledge Editing to Unstructured Data in Large Language Models

## Model Architecture
![](https://github.com/TrustedLLM/UnKE/blob/main/overview.png)
Comparison between UnKE and previous knowledge editing methods. Previous research assumed knowledge is stored as key-value pairs in local MLP layers, editing based on specific term positions like the subject. In contrast, UnKE views knowledge as stored in the global parameters of Transformer layers, considering the positions of all input tokens during editing. UnKE's editing process involves adjusting parameters in both MLP and attention layers, showing superior unstructured knowledge editing capabilities compared to prior methods.

## UnKEBench
UnKEBench is an unstructured knowledge editing benchmark. You can find the dataset at the following path.
```
./data/final_data_v3.json
```

## Requirements
The implementation of the causal attention mechanism is related to the version of the transformers library, so it is best to download the following version of the transformers library:
```
pip install transformers==4.40.1
```

## UnKE
UnKE is an efficient unstructured knowledge editing method. You can use it easily by following the steps below.

First, you can easily customize the editing parameters by directly modifying the configuration file config.py. This file allows you to make changes to various settings such as the base model used for editing, learning rate, batch size, and more. Currently we support two types of base models: llama2 and qwen1.5. The configuration file path is as follows,
```
./code/config.py
```
Next you can directly run unke.py to perform the editing process.
```
cd code
python3 unke_mmlu.py
```
It will output a json file containing the final inference results on UnKEBench. The inference results are then evaluated to obtain the final metrics.
```
python evaluate.py --file_path ../output/result.json --model_path ../model/all-MiniLM-L6-v2
```


