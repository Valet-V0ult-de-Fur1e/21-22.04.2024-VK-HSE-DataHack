import streamlit as st
from streamlit_tags import st_tags
import json
from stop_words import get_stop_words
import pymorphy3
import spacy
from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, Trainer
from datasets import Dataset

def load_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained("saved_model")
    model = AutoModel.from_pretrained("saved_model")
    print(model)
    return tokenizer, model


def preprocess(tokenizer, text):
    return tokenizer(text["text"], truncation=True, padding="max_length", max_length=128)


def predict(text, tokenizer, model):
    data = Dataset.from_dict({"text": [text], "label": [None]})
    tokenized_data = data.map(lambda x: preprocess(tokenizer, x))
    trainer = Trainer(model)
    return trainer.predict(tokenized_data).predictions


hide_decoration_bar_style = '''
    <style>
    header {visibility: hidden;}
    base="light"
    primaryColor="#0077ff"
    .reportview-container {
        background: green
    }
    .round {
        border-radius: 100px; /* Радиус скругления */
        border: 3px solid white; /* Параметры рамки */
    }
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
st.markdown(
        """
        <style>
        body {
            background-color: green;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<div class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #FFFFFF; border-width: 2px 0px 2px 0px; border-color: #e7e8ec; border-style: solid;">
  <div class="navbar-brand"  target="_blank">GGGGGGGGGGGGGG</div>
    <svg width="136" height="24" viewBox="0 0 575 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <g clip-path="url(#clip0_2_4)">
        <path d="M174.322 62.0419C174.322 69.6335 167.254 75.1309 157.306 75.1309H134.008V28.0105H156.521C166.207 28.0105 173.013 33.2461 173.013 40.576C173.013 45.5498 170.395 48.6911 166.469 50.5236C170.657 52.0943 174.322 56.2828 174.322 62.0419ZM144.217 36.6493V46.8587H156.259C159.924 46.8587 162.28 44.7644 162.28 41.8849C162.28 39.0053 159.662 36.6493 156.259 36.6493H144.217ZM157.306 66.4922C161.233 66.4922 164.113 64.1362 164.113 60.733C164.113 57.3299 161.495 54.9739 157.306 54.9739H144.217V66.4922H157.306Z" fill="black"/>
        <path d="M212.804 74.8691L196.574 56.2828H193.956V74.8691H183.484V28.0105H193.956V46.0733H196.574L212.28 28.0105H224.584L204.95 50.5236L226.416 74.8691H212.804Z" fill="black"/>
        <path d="M226.417 51.5709C226.417 37.4348 236.888 27.2253 251.547 27.2253C266.207 27.2253 276.678 37.4348 276.678 51.5709C276.678 65.707 266.207 75.9164 251.547 75.9164C236.888 75.9164 226.417 65.707 226.417 51.5709ZM265.684 51.5709C265.684 42.6704 259.924 36.6494 251.547 36.6494C243.17 36.6494 237.411 42.6704 237.411 51.5709C237.411 60.4714 243.17 66.4924 251.547 66.4924C259.924 66.4924 265.684 60.4714 265.684 51.5709Z" fill="black"/>
        <path d="M316.731 28.0105H327.202V74.8691H316.731V56.021H295.788V74.8691H285.317V28.0105H295.788V46.5969H316.731V28.0105Z" fill="black"/>
        <path d="M349.715 74.8691V37.4346H333.484V28.0105H376.155V37.4346H359.924V74.8691H349.715Z" fill="black"/>
        <path d="M421.181 48.6913V74.8693H413.589L411.495 67.5395C409.139 71.4662 404.165 75.9164 396.573 75.9164C387.149 75.9164 380.343 69.6337 380.343 60.995C380.343 52.3562 387.149 46.3353 401.024 46.3353H411.233C410.71 40.3143 407.568 36.1259 401.809 36.1259C397.097 36.1259 393.956 38.7437 392.385 41.6232L382.961 40.0526C385.317 31.9374 393.432 27.2253 402.333 27.2253C413.851 27.2253 421.181 35.0787 421.181 48.6913ZM410.971 54.4505H401.285C393.432 54.4505 390.814 57.0683 390.814 60.4714C390.814 64.3981 393.956 67.0159 399.191 67.0159C405.998 67.0159 410.971 62.0421 410.971 54.4505Z" fill="black"/>
        <path d="M461.495 74.8691L445.265 56.2828H442.647V74.8691H432.176V28.0105H442.647V46.0733H445.265L460.971 28.0105H473.275L453.642 50.5236L475.108 74.8691H461.495Z" fill="black"/>
        <path d="M493.956 74.8691V37.4346H477.725V28.0105H520.396V37.4346H504.165V74.8691H493.956Z" fill="black"/>
        <path d="M572.751 54.4505H535.055C536.102 61.7803 541.338 66.4924 549.191 66.4924C554.689 66.4924 558.877 64.1363 561.233 60.7332L570.919 62.3039C567.516 71.2044 558.354 75.9164 548.406 75.9164C534.27 75.9164 524.06 65.707 524.06 51.5709C524.06 37.4348 534.27 27.2253 548.406 27.2253C562.542 27.2253 572.751 37.4348 572.751 51.0473C573.013 52.3562 572.751 53.4034 572.751 54.4505ZM535.84 46.3353H561.757C559.924 40.5761 555.212 36.3876 548.668 36.3876C542.385 36.1259 537.411 40.3143 535.84 46.3353Z" fill="black"/>
        <path d="M0.5 48C0.5 25.3726 0.5 14.0589 7.52944 7.02944C14.5589 0 25.8726 0 48.5 0H52.5C75.1274 0 86.4411 0 93.4706 7.02944C100.5 14.0589 100.5 25.3726 100.5 48V52C100.5 74.6274 100.5 85.9411 93.4706 92.9706C86.4411 100 75.1274 100 52.5 100H48.5C25.8726 100 14.5589 100 7.52944 92.9706C0.5 85.9411 0.5 74.6274 0.5 52V48Z" fill="#0077FF"/>
        <path d="M53.7084 72.042C30.9167 72.042 17.9168 56.417 17.3752 30.417H28.7918C29.1668 49.5003 37.5833 57.5836 44.2499 59.2503V30.417H55.0003V46.8752C61.5836 46.1669 68.4995 38.667 70.8328 30.417H81.5831C79.7914 40.5837 72.2914 48.0836 66.9581 51.1669C72.2914 53.6669 80.8335 60.2086 84.0835 72.042H72.2498C69.7082 64.1253 63.3753 58.0003 55.0003 57.1669V72.042H53.7084Z" fill="white"/>
        </g>
        <defs>
        <clipPath id="clip0_2_4">
        <rect width="574" height="100" fill="white" transform="translate(0.5)"/>
        </clipPath>
        </defs>
    </svg>
    <div class="navbar-brand"  target="_blank">GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG</div>
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M3 5C3 4.29993 3 3.9499 3.13624 3.68251C3.25608 3.44731 3.44731 3.25608 3.68251 3.13624C3.9499 3 4.29993 3 5 3C5.70007 3 6.0501 3 6.31749 3.13624C6.55269 3.25608 6.74392 3.44731 6.86376 3.68251C7 3.9499 7 4.29993 7 5C7 5.70007 7 6.0501 6.86376 6.31749C6.74392 6.55269 6.55269 6.74392 6.31749 6.86376C6.0501 7 5.70007 7 5 7C4.29993 7 3.9499 7 3.68251 6.86376C3.44731 6.74392 3.25608 6.55269 3.13624 6.31749C3 6.0501 3 5.70007 3 5ZM3 12C3 11.2999 3 10.9499 3.13624 10.6825C3.25608 10.4473 3.44731 10.2561 3.68251 10.1362C3.9499 10 4.29993 10 5 10C5.70007 10 6.0501 10 6.31749 10.1362C6.55269 10.2561 6.74392 10.4473 6.86376 10.6825C7 10.9499 7 11.2999 7 12C7 12.7001 7 13.0501 6.86376 13.3175C6.74392 13.5527 6.55269 13.7439 6.31749 13.8638C6.0501 14 5.70007 14 5 14C4.29993 14 3.9499 14 3.68251 13.8638C3.44731 13.7439 3.25608 13.5527 3.13624 13.3175C3 13.0501 3 12.7001 3 12ZM3.13624 17.6825C3 17.9499 3 18.2999 3 19C3 19.7001 3 20.0501 3.13624 20.3175C3.25608 20.5527 3.44731 20.7439 3.68251 20.8638C3.9499 21 4.29993 21 5 21C5.70007 21 6.0501 21 6.31749 20.8638C6.55269 20.7439 6.74392 20.5527 6.86376 20.3175C7 20.0501 7 19.7001 7 19C7 18.2999 7 17.9499 6.86376 17.6825C6.74392 17.4473 6.55269 17.2561 6.31749 17.1362C6.0501 17 5.70007 17 5 17C4.29993 17 3.9499 17 3.68251 17.1362C3.44731 17.2561 3.25608 17.4473 3.13624 17.6825ZM10 5C10 4.29993 10 3.9499 10.1362 3.68251C10.2561 3.44731 10.4473 3.25608 10.6825 3.13624C10.9499 3 11.2999 3 12 3C12.7001 3 13.0501 3 13.3175 3.13624C13.5527 3.25608 13.7439 3.44731 13.8638 3.68251C14 3.9499 14 4.29993 14 5C14 5.70007 14 6.0501 13.8638 6.31749C13.7439 6.55269 13.5527 6.74392 13.3175 6.86376C13.0501 7 12.7001 7 12 7C11.2999 7 10.9499 7 10.6825 6.86376C10.4473 6.74392 10.2561 6.55269 10.1362 6.31749C10 6.0501 10 5.70007 10 5ZM10.1362 10.6825C10 10.9499 10 11.2999 10 12C10 12.7001 10 13.0501 10.1362 13.3175C10.2561 13.5527 10.4473 13.7439 10.6825 13.8638C10.9499 14 11.2999 14 12 14C12.7001 14 13.0501 14 13.3175 13.8638C13.5527 13.7439 13.7439 13.5527 13.8638 13.3175C14 13.0501 14 12.7001 14 12C14 11.2999 14 10.9499 13.8638 10.6825C13.7439 10.4473 13.5527 10.2561 13.3175 10.1362C13.0501 10 12.7001 10 12 10C11.2999 10 10.9499 10 10.6825 10.1362C10.4473 10.2561 10.2561 10.4473 10.1362 10.6825ZM10 19C10 18.2999 10 17.9499 10.1362 17.6825C10.2561 17.4473 10.4473 17.2561 10.6825 17.1362C10.9499 17 11.2999 17 12 17C12.7001 17 13.0501 17 13.3175 17.1362C13.5527 17.2561 13.7439 17.4473 13.8638 17.6825C14 17.9499 14 18.2999 14 19C14 19.7001 14 20.0501 13.8638 20.3175C13.7439 20.5527 13.5527 20.7439 13.3175 20.8638C13.0501 21 12.7001 21 12 21C11.2999 21 10.9499 21 10.6825 20.8638C10.4473 20.7439 10.2561 20.5527 10.1362 20.3175C10 20.0501 10 19.7001 10 19ZM17.1362 3.68251C17 3.9499 17 4.29993 17 5C17 5.70007 17 6.0501 17.1362 6.31749C17.2561 6.55269 17.4473 6.74392 17.6825 6.86376C17.9499 7 18.2999 7 19 7C19.7001 7 20.0501 7 20.3175 6.86376C20.5527 6.74392 20.7439 6.55269 20.8638 6.31749C21 6.0501 21 5.70007 21 5C21 4.29993 21 3.9499 20.8638 3.68251C20.7439 3.44731 20.5527 3.25608 20.3175 3.13624C20.0501 3 19.7001 3 19 3C18.2999 3 17.9499 3 17.6825 3.13624C17.4473 3.25608 17.2561 3.44731 17.1362 3.68251ZM17 12C17 11.2999 17 10.9499 17.1362 10.6825C17.2561 10.4473 17.4473 10.2561 17.6825 10.1362C17.9499 10 18.2999 10 19 10C19.7001 10 20.0501 10 20.3175 10.1362C20.5527 10.2561 20.7439 10.4473 20.8638 10.6825C21 10.9499 21 11.2999 21 12C21 12.7001 21 13.0501 20.8638 13.3175C20.7439 13.5527 20.5527 13.7439 20.3175 13.8638C20.0501 14 19.7001 14 19 14C18.2999 14 17.9499 14 17.6825 13.8638C17.4473 13.7439 17.2561 13.5527 17.1362 13.3175C17 13.0501 17 12.7001 17 12ZM17.1362 17.6825C17 17.9499 17 18.2999 17 19C17 19.7001 17 20.0501 17.1362 20.3175C17.2561 20.5527 17.4473 20.7439 17.6825 20.8638C17.9499 21 18.2999 21 19 21C19.7001 21 20.0501 21 20.3175 20.8638C20.5527 20.7439 20.7439 20.5527 20.8638 20.3175C21 20.0501 21 19.7001 21 19C21 18.2999 21 17.9499 20.8638 17.6825C20.7439 17.4473 20.5527 17.2561 20.3175 17.1362C20.0501 17 19.7001 17 19 17C18.2999 17 17.9499 17 17.6825 17.1362C17.4473 17.2561 17.2561 17.4473 17.1362 17.6825Z" fill="currentColor"></path></svg>
    <div class="navbar-brand"  target="_blank"></div>
    <img width="36px" height="36px" class="round" src="https://pp.userapi.com/60tZWMo4SmwcploUVl9XEt8ufnTTvDUmQ6Bj1g/mmv1pcj63C4.png" alt="Егор">
    <svg fill="none" height="8" viewBox="0 0 12 8" width="12" xmlns="http://www.w3.org/2000/svg"><path clip-rule="evenodd" d="M2.16 2.3a.75.75 0 0 1 1.05-.14L6 4.3l2.8-2.15a.75.75 0 1 1 .9 1.19l-3.24 2.5c-.27.2-.65.2-.92 0L2.3 3.35a.75.75 0 0 1-.13-1.05z" fill="currentColor" fill-rule="evenodd"></path></svg>
    <div class="collapse navbar-collapse" id="navbarNav">
  </div>
</div>
""", unsafe_allow_html=True)


#.image("VK Text Logo.svg", width=150)

def main():
    title = st.text_input('Заголовок (необязательно)')
    description = st.text_input('Краткое описание (необязательно)')
    txt = st.text_area(
    ":red[*]Текст статьи",
    ""
    )
 
    button = st.button("Обработать")

    if button or 'runned' in st.session_state:
        if len(str(txt).rstrip()) > 0:
            st.session_state['runned'] = txt
            main_tegs_list = ['MTag1', 'MTag2', 'MTag3', 'MTag4', 'MTag5']
            title = st.selectbox('Основной тег', main_tegs_list, predict(txt, load_tokenizer_model()))
            options= st_tags(
                label='Дополнительные теги',
                text='Нажмите enter, чтобы добавить новый тег',
                value=['Zero', 'One', 'Two']
                )
            
            with open('result.json', 'w') as fp:
                output = {
                    "main-tag": title,
                    "dop-tags": options
                }
                json.dump(output, fp)
            with open('result.json', 'rb') as file:
                dounload_button = st.download_button(
                    label="Отправить",
                    data=file,
                    mime='text/json',
                    file_name='result.json',
                )
            st.write(clearText(txt))
        else:
            st.error("Ошибка валидации!!! Не заполнены обязательные поля!")


    # st.header("Enter the statement that you want to analyze")
    # st.markdown("**Random Sentence:** A Few Good Men is a 1992 American legal drama film set in Boston directed by Rob Reiner and starring Tom Cruise, Jack Nicholson, and Demi Moore. The film revolves around the court-martial of two U.S. Marines charged with the murder of a fellow Marine and the tribulations of their lawyers as they prepare a case to defend their clients.")
    
    # text_input = st.text_area("Enter sentence")
    
    # ner = en_core_web_sm.load()
    # doc = ner(str(text_input))
    
    # spacy_streamlit.visualize_ner(doc, labels=ner.get_pipe('ner').labels)

def clearText(text):
    doc = st.session_state['nlp'](text)
    keywords_freq = [token.text for token in doc if not token.is_stop and token.is_alpha]
    filtered_words = []
    for word in keywords_freq:
        if word not in get_stop_words('russian'):
            outword = word
            try:
                fdf =  st.session_state['morph'].parse(outword)[0]
                outword = fdf.inflect({'nomn', 'sing'}).word
            except:
                pass
            filtered_words.append(outword)
    return st.session_state['KeyBERT'].extract_keywords(
        filtered_words,
        top_n=10,
        keyphrase_ngram_range=(1, 2),
    )



if __name__ == "__main__":
    if 'morph' in st.session_state:
        pass
    else:
        st.session_state['nlp'] = spacy.load("ru_core_news_sm")
        st.session_state['morph'] = pymorphy3.MorphAnalyzer()
        st.session_state['KeyBERT'] = KeyBERT(model="cointegrated/rubert-tiny2")
    main()
