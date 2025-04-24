import streamlit as st


# Para ejecutar el modelo en Streamlit
# streamlit run forecasting-bluesky-app.py

# -------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------

st.set_page_config(page_title='FORECASTING BLUESKY', 
                   page_icon='🦋', 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items={'About':'This app is a model deployment for a Data Science Bootcamp final project by Camino de Pablos. For more info about the project or the app, do not heasitate to contact me on caminodepablos@gmail.com. GitHub Repository: https://github.com/caminodepablos/forecasting_bluesky'})


pages = {
    "About this project": [
        st.Page("page1.py", title="Forecasting Bluesky"),
        st.Page("page2.py", title="Bluesky"),
        st.Page("page3.py", title="About Me")
    ],
    "Predict": [
        st.Page("page4.py", title="News Impact on Activity"),
        st.Page("page5.py", title="Tomorrow Predictions")
    ],
}

pg = st.navigation(pages)
pg.run()


