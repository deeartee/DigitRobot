# Digit classifier v4

CNN для цифр 0–9, 28×28.

## Файлы
- `config.py`       — параметры запуска
- `synth_digits.py` — синтез датасета
- `conv_model.py`   — модель `DigitConv`
- `runner.py`       — обучение → `model.pt`
- `app.py`          — предсказание

## Команды
    pip install -r requirements.txt
    python app.py train
    python app.py predict --demo
    python app.py predict img.png