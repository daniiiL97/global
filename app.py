import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from bs4 import BeautifulSoup
import torch

# Загрузка предобученной модели и токенизатора
tokenizer = GPT2Tokenizer.from_pretrained("danik97/global-generator-ai")
model = GPT2LMHeadModel.from_pretrained("danik97/global-generator-ai")

# Заголовок приложения
st.title("Глобал ГЕЙнерация")

# Поле для ввода текста
text_input = st.text_input("Введите начало текста для генерации:")

# Ползунок для выбора температуры
temperature = st.slider("Выберите температуру:", min_value=0.01, max_value=2.0, step=0.1, value=0.9)

# Ползунок для выбора количества слов
max_words = st.slider("Выберите количество слов:", min_value=10, max_value=200, step=5, value=50)

# Секция с информацией о модели
st.sidebar.title("О МОДЕЛИ:")
st.sidebar.markdown("""
- **Была дообучена большая языковая модель на базе GPT-2 - sberbank-ai/rugpt3small_based_on_gpt2.**
\n
- **В качестве входных данных был собран датасет объемом 16 000 строк из беседы NSDPRSSD.**
""")

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

    # Отображение сгенерированного текста
    st.subheader("Сгенерированный текст:")
    st.write(generated_text)

# Пояснение о температуре
st.sidebar.markdown("""
### О ТЕМПЕРАТУРЕ:
- Параметр температуры влияет на случайность генерации текста.
- Чем выше значение, тем более случайным и разнообразным и «креативным» будет текст.
- Чем ниже значение, тем более детерминированным и предсказуемым будет текст.
""")

# Дополнительное пояснение о возможной ошибке
st.sidebar.markdown("""
### ПРО ОШИБКУ:
* Если вы введете большой промт и поставите минимальную длину, то выдаст ошибку.
* Для решения нужно просто увеличить ползунок количества слов и ошибка пропадет.
""")
