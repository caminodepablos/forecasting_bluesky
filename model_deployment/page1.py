import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# -------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------

st.title(':gray[FORECASTING BLUESKY]')
st.subheader(':gray[Predicción del crecimiento de usuarios a través del análisis de actividad y el impacto de noticias]')

st.subheader(':rainbow-background[:gray[Hipótesis: existe una correlación entre los cambios políticos y sociales y el aumento de usuarios en Bluesky.]]')

st.write("Contenido del directorio actual:")
st.write(os.listdir('.'))

# -------------------------------------------------------
# PLOTS
# -------------------------------------------------------

# Dataframe para los plots
df = pd.read_csv('final_dataset_bsky_news.csv')

# Definición de los tabs
tab1, tab2 = st.tabs(["New Users", "Activity"])

# Gráfico de nuevos usuarios por día
with tab1:
    st.area_chart(df, x= 'date', y = ['new_users'], color=["#FF0000"], y_label='Bluesky New Users')
    with st.expander("Detalle"):
        st.write('''
            Este gráfico muestra los nuevos usuarios por día en la red social Bluesky.
            Los picos de crecimiento de usuarios coinciden con fechas muy significativas:
            - **feb 6, 2024**: Bluesky se abre a todo el público.
            - **sept 19, 2024**: X anuncia que se plantea cobrar a todos sus usuarios.
            - **nov 5, 2024**: Trump gana las elecciones de EEUU.
            - **jan 20, 2025**: Trump entra en la Casa Blanca.
        ''')
        st.caption('Datos de nuevos usuarios por día desde el 8 de mayo de 2023 hasta el 7 de abril de 2025')
        st.caption('Datos obtenidos de [Bluesky Stats, by Jaz](https://bsky.jazco.dev/stats)')

# Gráfico de resumen de actvidad diaria
with tab2:
    st.area_chart(df, x= 'date', y = ['num_likes'], color=["#0000FF"], y_label='Bluesky Activity')
    with st.expander("Detalle"):
        st.write('''
            Este gráfico muestra la actividad diaria (en concreto el número de likes diarios) en la red social Bluesky.
            Los picos de crecimiento de usuarios coinciden con fechas muy significativas:
            - **feb 6, 2024**: Bluesky se abre a todo el público.
            - **sept 19, 2024**: X anuncia que se plantea cobrar a todos sus usuarios.
            - **nov 5, 2024**: Trump gana las elecciones de EEUU.
            - **jan 20, 2025**: Trump entra en la Casa Blanca.
        ''')
        st.caption('Datos de nuevos usuarios por día desde el 8 de mayo de 2023 hasta el 7 de abril de 2025')
        st.caption('Datos obtenidos de [Bluesky Stats, by Jaz](https://bsky.jazco.dev/stats)')

