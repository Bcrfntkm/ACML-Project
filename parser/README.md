# parser/

Этот каталог содержит **ReactionMiner** как git submodule — внешний PDF-парсер,
разработанный для извлечения текста из статей по органической химии.

Нам нужен только модуль `pdf2text/`. Остальные части репозитория
(`extraction/`, `segmentation/`) не используются в нашем пайплайне.

## Цитирование

Если используешь этот парсер в публикации, цитируй оригинальную работу:

```
@inproceedings{zhong2023reactionminer,
  title     = {ReactionMiner: An Integrated System for Chemical Reaction Extraction
               from Organic Chemistry Literature},
  author    = {Zhong, Ming and others},
  booktitle = {Proceedings of EMNLP 2023 (System Demonstrations)},
  year      = {2023},
  url       = {https://github.com/maszhongming/ReactionMiner}
}
```

SymbolScraper (Java-компонент внутри `pdf2text/`) цитируется отдельно:

```
@inproceedings{zanibbi2021symbolscraper,
  title  = {SymbolScraper: A PDF Parser for Mathematical Formulas},
  author = {Zanibbi, Richard and others},
  year   = {2021}
}
```

## Подключение в свой git-репозиторий

Когда будешь инициализировать свой репо (`git init` в workspace root):

```bash
# 1. Зарегистрировать как submodule (вместо простого клона)
git submodule add https://github.com/maszhongming/ReactionMiner.git Science/parser/ReactionMiner

# 2. Инициализировать вложенный submodule (SymbolScraper)
git submodule update --init --recursive

# 3. Зафиксировать конкретный коммит (воспроизводимость)
cd Science/parser/ReactionMiner
git log --oneline -1   # запиши этот хэш в документацию
cd -
git add .gitmodules Science/parser/ReactionMiner
git commit -m "add ReactionMiner as git submodule"
```

Если репо уже клонировано вручную (текущая ситуация):

```bash
# Удали папку и переклонируй как submodule
rm -rf Science/parser/ReactionMiner
git submodule add https://github.com/maszhongming/ReactionMiner.git Science/parser/ReactionMiner
git submodule update --init --recursive
```

## Что нужно собрать

```bash
# Собрать SymbolScraper (Java 1.8 JDK + Maven 3.x)
cd Science/parser/ReactionMiner/pdf2text/SymbolScraper
mvn package
cd -
```

## Что игнорировать

Следующие директории содержат промежуточные файлы парсинга и не должны
попадать в git (уже добавлены в `Science/.gitignore`):

```
parser/ReactionMiner/pdf2text/results/
parser/ReactionMiner/pdf2text/parsed_raw/
parser/ReactionMiner/pdf2text/xmlFiles/
```
