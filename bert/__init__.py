"""BERT fine-tuning module for chemical NER and property role extraction.

Scripts:
  convert_to_bio.py  — LLM annotations → BIO token labels (JSONL)
  train.py           — fine-tune BERT/SciBERT/BioBERT via HuggingFace Trainer
  evaluate.py        — seqeval precision/recall/F1 per label

Configs (bert/configs/):
  bert_001.yaml  — bert-base-uncased  (NER, general domain baseline)
  bert_002.yaml  — scibert            (NER, scientific text)
  bert_003.yaml  — biobert            (NER, biomedical text)
  bert_004.yaml  — scibert            (property role labeling, 5-role schema)

Workflow:
  1. Accumulate LLM annotations on 5+ papers  →  data/annotations/*.json
  2. python bert/convert_to_bio.py --input ... --output data/training/ner/ --task ner --split
  3. python bert/train.py bert/configs/bert_002.yaml
  4. python bert/evaluate.py --model models/bert_002/best_model --test data/training/ner/test.jsonl
"""
