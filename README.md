# ml-tinkoff
n-gram language model<br />
Параметры `train.py`:<br />
`--input-dir`  путь к директории, в которой лежит коллекция документов. Если данный аргумент не задан, то тексты вводятся из stdin.<br />
`--model`  путь к файлу, в который сохраняется модель.<br />
`--seq-len` длина префикса.<br />

Параметры `generate.py`:<br />
`--model`  путь к файлу, из которого загружается модель.<br />
`--prefix` необязательный аргумент. Начало предложения (одно или несколько слов). Если не указано, выбираем начальное слово случайно из всех слов.<br />
`--length` длина генерируемой последовательности.
