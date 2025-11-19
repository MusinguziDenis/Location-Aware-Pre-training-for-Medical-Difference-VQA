# Location-Aware Pre-training for Medical Difference VQA

This is the official release of the training code for the Location-Aware Pre-training for Medical Difference VQA paper.

## Getting Started
### Installation
#### Prepare the code and the environment

Git clone our repository, create a python environment and activate it via the command

```bash
git clone https:...git
cd "Location-Aware Pre-training for Medical Difference VQA"
conda env create -f environment.yml
conda activate lloca
```

#### Training
The training of the model contains two stages

##### 1. First pretraining stage
In the first stage, the image encoder is trained on image-text pairs derived from the [Chest ImaGenome Dataset](https://physionet.org/content/chest-imagenome/1.0.0/) scene graphs to improve its ability to extract fine-grained features.
To download and prepare the datasets, you will need access to the datasets through [Physionet](https://physionet.org). You need to be a credentialed user to access the datasets. To launch the first stage training, run the following command. In our experiments, we use a single L40 GPU. 
```bash
python LocCa/main.py
```

##### 2. Second finetuning stage
In the second stage, we combine the model with a GPT2 medium decoder and finetune it for the Medical Difference VQA task. To access the [Medical-Diff-VQA Dataset](https://physionet.org/content/medical-diff-vqa/1.0.1/) on [Physionet](https://physionet.org), you need to be a credentialed user. To launch finetuning, you need to specificy the checkpoint of the pre-trained vision encoder in the main script. Run the following command to finetune the model. In our experiments, we use a single L40 GPU.

```bash
python main.py
```

After the finetuning, you can run inference on the model using the following code in the inference script.
```bash
python inference.py --model-checkpoint --prompt "What has changed compared to the reference image?" --main-image-path "path-to-main-image" --ref-image-path "path-to-ref-image" --max-tokens-to-generate 64 --temperature 0.2 --top-k 20
```
If you're using Location-Aware Pretraining for Medical Difference VQA in your research or applications, please cit using this BibTex
```
@article{musinguzi2025locca,
    title={Location-Aware Pre-training for Medical Difference VQA},
    author={Musinguzi, Denis and Mitra, Prasenjit}},
    journal={arXiv preprint arXiv:}
    year={2025}
```

#### License
This repository is under the MIT License - see the [LICENSE](LICENSE) file for details.