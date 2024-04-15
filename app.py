import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from bs4 import BeautifulSoup
import torch
import nltk.data
import re

# Загрузка модели и токенизатора с использованием кеширования
@st.cache(allow_output_mutation=True)
def load_model():
    return GPT2LMHeadModel.from_pretrained("danik97/global-generator-ai")

@st.cache(allow_output_mutation=True)
def load_tokenizer():
    return GPT2Tokenizer.from_pretrained("danik97/global-generator-ai")

# Загрузка модели и токенизатора
tokenizer = load_tokenizer()
model = load_model()

# Заголовок приложения
st.title("Глобал ГЕЙнерация")

# Поле для ввода текста
text_input = st.text_area("Введите начало текста для генерации:")

# Ползунок для выбора температуры
temperature = st.slider("Выберите температуру:", min_value=0.01, max_value=2.0, step=0.1, value=0.9)

# Ползунок для выбора количества слов
max_words = st.slider("Выберите количество слов:", min_value=10, max_value=200, step=5, value=50)

# Регулярные выражения для удаления URL и временных меток
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
time_pattern = r'\b(?:вчера|сегодня)\s+\d{1,2}\s*:\s*\d{2}\s*\w{2}\s+\[\w,\]+\s+\d{1,2}/\d{1,2}\b'

# Если пользователь ввел текст, то выполняется генерация текста
if text_input:
    text_without_html = BeautifulSoup(text_input, "html.parser").get_text()
    inputs = tokenizer.encode(text_without_html, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
        model.to('cuda')

    # Генерация текста с использованием параметров, выбранных пользователем
    output = model.generate(inputs,
                            max_length=max_words,
                            repetition_penalty=10.0,
                            do_sample=True,
                            top_k=5,
                            top_p=0.95,
                            temperature=temperature,
                            no_repeat_ngram_size=2,
                            pad_token_id=tokenizer.eos_token_id)

    # Преобразование сгенерированного текста в строку
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Удаление ссылок и временных меток из сгенерированного текста
    generated_text_cleaned = re.sub(url_pattern, '', generated_text)
    generated_text_cleaned = re.sub(time_pattern, '', generated_text_cleaned)

    # Разбиение сгенерированного текста на предложения с использованием NLTK
    tokenizer_nltk = nltk.data.load('tokenizers/punkt/russian.pickle')
    sentences = tokenizer_nltk.tokenize(generated_text_cleaned)

    # Соединение предложений в единый текст с добавлением точек и знаков препинания
    full_text = ' '.join([sentence.strip() + '.' for sentence in sentences if sentence.strip()])

    # Отображение сгенерированного текста с точками и знаками препинания
    st.subheader("Сгенерированный текст с точками и знаками препинания:")
    st.write(full_text)

# Сайдбар с дополнительной информацией о модели и температуре
st.sidebar.title("О МОДЕЛИ:")
st.sidebar.markdown("""
- **Была дообучена большая языковая модель на базе GPT-2 - sberbank-ai/rugpt3small_based_on_gpt2.**
\n
- **В качестве входных данных был собран датасет объемом 16 000 строк из беседы NSDPRSSD.**
""")

st.sidebar.markdown("""
### О ТЕМПЕРАТУРЕ:
- Параметр температуры влияет на случайность генерации текста.
- Чем выше значение, тем более случайным и разнообразным будет текст.
- Чем ниже значение, тем более детерминированным и предсказуемым будет текст.
""")
