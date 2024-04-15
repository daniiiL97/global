import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from bs4 import BeautifulSoup
import torch
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
text_input = st.text_input("Введите начало текста для генерации:")

# Ползунок для выбора температуры
temperature = st.slider("Выберите температуру:", min_value=0.01, max_value=2.0, step=0.1, value=0.9)

# Ползунок для выбора количества слов
max_words = st.slider("Выберите количество слов:", min_value=10, max_value=200, step=5, value=50)

# Регулярные выражения для удаления URL
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

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

    # Удаление ссылок из сгенерированного текста
    generated_text_cleaned = re.sub(url_pattern, '', generated_text)

    # Разделение сгенерированного текста на предложения с учетом существующих знаков препинания
    sentences = split_into_sentences(generated_text_cleaned)

    # Формирование окончательного текста с правильным форматом предложений
    formatted_text = ' '.join(sentences)

    # Отображение сгенерированного текста
    st.subheader("Сгенерированный текст:")
    st.write(formatted_text)

# Функция для разделения текста на предложения с учетом существующих знаков препинания
def split_into_sentences(text):
    sentences = []
    current_sentence = []

    for char in text:
        current_sentence.append(char)

        if char in '.!?':
            sentences.append(''.join(current_sentence).strip())
            current_sentence = []

    if current_sentence:
        sentences.append(''.join(current_sentence).strip())

    return sentences
