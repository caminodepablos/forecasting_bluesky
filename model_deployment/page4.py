import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from impact_score import impact_score_pipeline


# -------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------

st.title(':gray[NEWS IMPACT ON ACTIVITY]')
st.subheader(':gray[Predicción del impacto que tienen las noticias en el aumento de la actividad en Bluesky ***(Impact Score)***]')

# -------------------------------------------------------
# MODELO Y DATASET
# -------------------------------------------------------

impact_score_model = joblib.load('impact_score_xgbclass_model.pkl')


# -------------------------------------------------------
# FUNCIONES
# -------------------------------------------------------

def text_cleaning(text):
    '''
    Basic text cleaning (English):
    - NaN: ''
    - Lowercase
    - Eliminate punctuations
    - Space cleaning
    '''
    if pd.isnull(text):
        return ""
    # Lowercase
    text = text.lower()
    # Eliminate punctuations
    text = re.sub(r"[^a-záéíóúüñ\s]", "", text)
    # Space cleaning
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'\[\d+\]', '', text)
    return text

# -------------------------------------------------------
# FORMULARIO
# -------------------------------------------------------

# Features que le pedimos al usuario
st.markdown('Introduce los headlines de hoy: ')
st.caption(f"Las últimas noticias se recogieron el 23/04/2025")

with st.form("my_form"):
    #ia = st.text_input("**Noticias sobre IA**")
    #ia = text_cleaning(ia)
    #crime = st.text_input("**Noticias sobre Crime**")
    #crime = text_cleaning(crime)
    #education = st.text_input("**Noticias sobre Education**")
    #education = text_cleaning(education)
    elections = st.text_input("Noticias sobre **elecciones**")
    elections = text_cleaning(elections)
    #entertainment = st.text_input("**Noticias sobre entertainment**")
    #entertainment = text_cleaning(entertainment)
    #environment = st.text_input("**Noticias sobre environment**")
    #environment = text_cleaning(environment)
    finance = st.text_input("Noticias sobre **economía**")
    finance = text_cleaning(finance)
    #health = st.text_input("**Noticias sobre health**")
    #health = text_cleaning(health)
    international_relations = st.text_input("Noticias sobre **relaciones internacionales**")
    international_relations = text_cleaning(international_relations)
    #immigration = st.text_input("**Noticias sobre immigration**")
    #immigration = text_cleaning(immigration)
    #pandemics = st.text_input("**Noticias sobre pandemics**")
    #pandemics = text_cleaning(pandemics)
    politics = st.text_input("Noticias sobre **política**")
    politics = text_cleaning(politics)
    #protests = st.text_input("**Noticias sobre protests**")
    #protests = text_cleaning(protests)
    science = st.text_input("**Noticias sobre **ciencia**")
    science = text_cleaning(science)
    #sports = st.text_input("**Noticias sobre sports**")
    #sports = text_cleaning(sports)
    #technology = st.text_input("**Noticias sobre technology**")
    #technology = text_cleaning(technology)
    #videogames = st.text_input("**Noticias sobre videogames**")
    #videogames = text_cleaning(videogames)
    #war = st.text_input("**Noticias sobre war**")
    #war = text_cleaning(war)
     
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.write(':red[Datos actualizados]')

# Diccionario con todos los inputs
new_data = {
    'artificial intelligence': [''],
    'crime': [''],
    'education': [''],
    'elections': [elections],
    'entertainment': [''],
    'environment': [''],
    'finance': [finance],
    'health': [''],
    'immigration': [''],
    'international relations': [international_relations],
    'pandemics': [''],
    'politics': [politics],
    'protests': [''],
    'science': [science],
    'sports': [''],
    'technology': [''],
    'videogames': [''],
    'war': [''],
}

df = pd.DataFrame(new_data)

if not df.replace('', np.nan).dropna(how='all').empty:
    impact_score_array = impact_score_pipeline.fit_transform(df)
else:
    impact_score_array = None
    st.caption(':red[Introduce al menos un titular para obtener tu impact score]')


# -------------------------------------------------------
# CALCULATE IMPACT SCORE
# -------------------------------------------------------

st.divider()

pred_score = 0

if submitted and impact_score_array is not None:
    st.divider()
    pred_score = 0

    if st.button("Predict News Impact Score 🚀", type='primary', use_container_width=True):
        try:
            pred_score = impact_score_model.predict_proba(impact_score_array)[:, 1][-1]
        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")

    st.subheader("Impact Score para las noticias de hoy:")
    st.header(f':rainbow-background[**{round(pred_score, 2)}**]')

