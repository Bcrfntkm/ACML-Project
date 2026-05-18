# baselines/

Скрипты для сравнения нашего LLM-пайплайна с готовыми инструментами химического NER.
Все скрипты принимают путь к PDF и выдают JSON в том же формате, что и основной пайплайн:
`[{"text": ..., "label": ..., "start": ..., "end": ...}, ...]`

## Бейзлайны

| Инструмент | Тип | Установка | Скрипт |
|---|---|---|---|
| spaCy `en_core_web_sm` | ML, общий домен | уже в venv | `run_spacy.py` |
| GLiNER `gliner_mediumv2.1` | zero-shot NER | `pip install gliner` | `run_gliner.py` |
| ChemDataExtractor 2 | rule-based + ML, химия | `pip install chemdataextractor2` | `run_chemdataextractor.py` |

`compare.py` — сравнение выходов нескольких бейзлайнов между собой или с эталоном.

## Быстрый старт

```bash
source Science/agent-venv/bin/activate

# spaCy (работает сразу)
python Science/baselines/run_spacy.py Science/data/pdfs/paper.pdf -o Science/data/annotations/spacy_out.json

# GLiNER (нужен pip install gliner)
pip install gliner
python Science/baselines/run_gliner.py Science/data/pdfs/paper.pdf -o Science/data/annotations/gliner_out.json

# ChemDataExtractor (нужен pip install + download models)
pip install chemdataextractor2
python -m chemdataextractor download
python Science/baselines/run_chemdataextractor.py Science/data/pdfs/paper.pdf -o Science/data/annotations/cde_out.json

# Сравнение двух файлов с аннотациями
python Science/baselines/compare.py \
    --pred Science/data/annotations/gliner_out.json \
    --ref  Science/data/annotations/copper_acetate_v2.json
```

## Формат сравнения

`compare.py` считает:
- **Exact match** — span и метка совпадают точно
- **Partial match** — текст пересекается, метка совпадает
- Precision / Recall / F1 по каждой метке и суммарно

## Литература

- **ChemDataExtractor 2.0** — Mavracic et al., *J. Chem. Inf. Model.* 2021, 61(9), 4280
- **GLiNER** — Zaratiana et al., *arXiv:2311.08526*, 2023
- **ChEMU benchmark** (оценка химического NER) — Nguyen et al., *ECIR 2020*
