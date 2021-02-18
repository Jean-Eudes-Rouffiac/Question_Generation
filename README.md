# Question Generation

L'objectif de ce projet est de générer automatiquement des questions à partir de contextes. Le modèle utilisé est est un Text-to-Text Transfer Transformer (T5) développé par Google. Il a été entraîné sur un large Data Set (C4) et peut être fine tuné rapidement et facilement pour n'importe quelle tâche de Text-to-Text (summarization, question answering, question generation ...). Pour utiliser des données textuelles en français, le modèle [t5-base-multi-fr-wiki-news](https://huggingface.co/airKlizz/t5-base-multi-fr-wiki-news) est utilisé. On peut alors utiliser ce modèle pré entraîné dans n'importe quelle tâche de textto-text avec du vocabulaire français. Repo basé et adapté à partir de Suraj Patil (https://github.com/patil-suraj/question_generation) et Joachim Dublineau (https://github.com/joachim-dublineau).

Pour plus d'informations concernant les transformers : [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf).  
Article présentant le modèle T5 : [Exploring the Limits of Transfer Learning with a UnifiedText-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf).  
Autre article très intéressant sur la génération de questions : [Transformer-based End-to-End Question Generation](https://arxiv.org/pdf/2005.01107v1.pdf).  

## Entraînement avec FQUAD et PIAD

### Génération des data sets train et valid 

```bash
python data/fquad_multitask/fquad_multitask.py --path_to_data data/fquad_multitask/
```

### Data preprocessing
 
```bash
python prepare_data.py --task e2e_qg --valid_for_qg_only --model_type t5 --dataset_path data/ \
--qg_format highlight_qg_format --max_source_length 512 --max_target_length 32 --train_file_name train.csv \
--valid_file_name valid.csv --train_file_name_cache train_data_e2e_qg_t5.pt --valid_file_name_cache valid_data_e2e_qg_t5.pt
```


### Training

Les informations concernant les entraînements : https://wandb.ai/jedrouffiac/generation_question/?workspace=user-jedrouffiac.

```bash
python train.py --model_name_or_path airKlizz/t5-base-multi-fr-wiki-news --model_type t5 \
--tokenizer_name_or_path t5_qg_tokenizer --output_dir ../t5-fr-e2e-hl/run0 --train_file_path data/train_data_e2e_qg_t5.pt \
--valid_file_path data/valid_data_e2e_qg_t5.pt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 \
--gradient_accumulation_steps 8 --learning_rate 1e-4 --num_train_epochs 5 --seed 42 --do_train --do_eval \
--evaluate_during_training --logging_steps 20
```

### Génération de questions

```bash
python generate.py  --file_data clean_fquad_valid_for_generate.csv --output_dir results --file_name question_generation_predictions \
--checkpoint_path t5-fr-e2e-hl/run0 --tokenizer_name_or_path t5_qg_tokenizer 
```

## Credits:
Credits to Suraj Patil (https://github.com/patil-suraj/question_generation) and Joachim Dublineau (https://github.com/joachim-dublineau)
