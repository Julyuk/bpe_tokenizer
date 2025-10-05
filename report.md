# Звіт: Власний BPE токенізатор (byte-fallback, EN+UK)

## Навчальний датасет
- Файли (не з валідації):


## Розмір словника
- Вибраний розмір словника: **3209**
- Кількість мерджів: **2950** (враховуючи базу 259)

## Код навчання
```bash
python scripts/train_bpe.py   --train-files    --vocab-size 3209   --min-pair-count 2   --model-out bpe_model.json
```
```python
from src.bpetok.bpe import BPETrainer
texts = [open(p, 'r', encoding='utf-8').read() for p in []]
trainer = BPETrainer(vocab_size=3209, min_pair_count=2)
tok = trainer.train(texts)
tok.save('bpe_model.json')
```


## Код токенізації і детокенізації
```python
from src.bpetok.bpe import BPETokenizer

# Load trained model
tok = BPETokenizer.load('bpe_model.json')

# Tokenize and detokenize
text = "Hello, world! Привіт, світе!"
ids = tok.encode(text)
print(ids)
print(tok.decode(ids))
```


## Метрики на валідаційних наборах
| файл | tokens | words | bytes | chars | tokens/word | bytes/token | chars/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| val_en.txt | 17716 | 7820 | 51402 | 51368 | 2.2655 | 2.9014 | 2.8995 |
| val_uk.txt | 7338 | 2099 | 31550 | 17339 | 3.4960 | 4.2995 | 2.3629 |

**Сукупно:** tokens=25054, words=9919, bytes=82952, chars=68707; tokens/word=2.5259, bytes/token=3.3109, chars/token=2.7424


## Нотатки
- Базовою абеткою є 256 байтів → byte-fallback гарантує покриття будь-яких символів.
- Моделі/корпуси для валідації (Brown/Laws) не використовуються у навчанні.
- Метрики: fertility (tokens/word), bytes/token, chars/token.
