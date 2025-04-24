import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------

st.title(':gray[TOMORROW PREDICTIONS]')
st.subheader(':gray[Predicción del número de nuevos usuarios y nivel de actividad que habrá mañana en Bluesky]')

# -------------------------------------------------------
# MODELO Y DATASET
# -------------------------------------------------------

# Modelos
modelo = joblib.load("linear_reg_bsky.pkl")
logistic_class_model = joblib.load("logistic_reg_bsky.pkl")

# Dataset
data = pd.read_csv('daily_dataset_model_features_reg.csv')
data_original = pd.read_csv('final_dataset_bsky_news.csv')
num_data = data_original._get_numeric_data()

# -------------------------------------------------------
# FUNCIONES
# -------------------------------------------------------

def activity_score_pca(df, activity_cols = ['num_likes', 'num_posts', 'num_images', 'num_follows', 'num_blocks']):
    
    # mean 0, std 1
    scaler = StandardScaler()
    norm_activity = scaler.fit_transform(df[activity_cols])
    
    # PCA
    pca = PCA(n_components=1)
    df['activity_score_pca'] = pca.fit_transform(norm_activity)
    
    # Activity cols dropping
    df.drop(columns = (activity_cols), inplace=True)

    return df

def temp_columns(df, columns = ['users', 'new_users', 'activity_score_pca'], window = 7):
    '''
    Receives full dataframe (must have date colum) and list of columns to create (optional). Returns full df with new temp columns:
    - Lags (yesterday data for users, new users and activity score)
    - Rolling Mean (7 day rolling mean data for new users and activity score)
    - Diff (diff today vs yesterday for new users and activity score)
    - NaN = 0
    '''
    
    # Lags
    df['users_lag1'] = df['users'].shift(1)
    df['new_users_lag1'] = df['new_users'].shift(1)
    df['activity_lag1'] = df['activity_score_pca'].shift(1)
    
    # Rolling mean
    df['new_users_ma7'] = df['new_users'].rolling(window=window).mean()
    df['activity_ma7'] = df['activity_score_pca'].rolling(window=window).mean()
    
    # Diff
    df['new_users_diff'] = df['new_users'].diff()
    df['activity_diff'] = df['activity_score_pca'].diff()

    # Fills NaN with 0
    df = df.fillna(0)

    return df

def categorize_quantile(df, var, quantile_num = [0.25, 0.5, 0.75]):
    '''
    Receives a dataframe column (var) and a list of 3 quantile (optional).
    Returns new column with categorization applied
    '''

    quantile = df[var].quantile(quantile_num)
    
    def categorize_score(score):
        if score <= quantile[quantile_num[0]]:
            return 0 # low
        elif score <= quantile[quantile_num[1]]:
            return 1 # medium
        elif score <= quantile[quantile_num[2]]:
            return 2 # high
        else:
            return 3 # very high
    
    return df[var].apply(categorize_score)

# -------------------------------------------------------
# FORMULARIO
# -------------------------------------------------------

# Features que le pedimos al usuario
st.markdown('Introduce los datos de hoy: ')
st.caption(f'Últimos datos recogidos a fecha {data_original['date'].iloc[-1]}')
with st.form("my_form"):
    col1, col2 = st.columns(2)
    with col1:
        new_tot_users = st.number_input("Total de usuarios", value=round(data_original['users'].max()), placeholder="Type a number...")
        st.caption(f"El último dato recogido es {round(data_original['users'].max())}")
        
        new_num_likes = st.number_input("Total de likes", value=data_original['num_likes'].max(), placeholder="Type a number...")
        st.caption(f"El último dato recogido es {round(data_original['num_likes'].max())}")
        
        new_num_posts = st.number_input("Total de posts", value=data_original['num_posts'].max(), placeholder="Type a number...")
        st.caption(f"El último dato recogido es {round(data_original['num_posts'].max())}")
    with col2:
        new_num_images = st.number_input("Total de imágenes", value=data_original['num_images'].max(), placeholder="Type a number...")
        st.caption(f"El último dato recogido es {round(data_original['num_images'].max())}")
        
        new_num_follows = st.number_input("Total de follows", value=data_original['num_follows'].max(), placeholder="Type a number...")
        st.caption(f"El último dato recogido es {round(data_original['num_follows'].max())}")
        
        new_num_blocks = st.number_input("Total de blocks", value=data_original['num_blocks'].max(), placeholder="Type a number...")
        st.caption(f"El último dato recogido es {round(data_original['num_blocks'].max())}")
        
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write(':red[Datos actualizados]')

# -------------------------------------------------------
# ACTUALIZAR DATASET
# -------------------------------------------------------

# Actualizamos el dataframe con los nuevos datos introducidos por el usuario
new_data = {
    'users': [new_tot_users],
    'num_likes': [new_num_likes],
    'num_posts': [new_num_posts],
    'num_images': [new_num_images],
    'num_follows': [new_num_follows],
    'num_blocks': [new_num_blocks]
}

new_data_df = pd.DataFrame(new_data, columns=[
    'users', 'new_users', 'num_likes', 'num_posts', 'num_images', 'num_follows', 'num_blocks'
])

num_data = pd.concat([num_data, new_data_df], ignore_index=True)

# Creamos el nuevo dato de new_users
num_data['new_users'] = num_data['users'].diff(1)
num_data['new_users'] = num_data['new_users'].fillna(0)

# Creamos la columna de PCA
num_data = activity_score_pca(num_data)

# Creamos los lags
num_data = temp_columns(num_data)

# Renombramos tot_users por si acaso
num_data= num_data.rename(columns={'users':'tot_users'})

# Creamos columna target
num_data['target'] = num_data['new_users'].shift(-1)
num_data['target'] = num_data['target'].fillna(0)

# Rellenamos el impact score de momento con lo que tenemos
num_data['impact_score'] = data['impact_score']
num_data['impact_score'] = num_data['impact_score'].fillna(0)

num_data_class = num_data.copy()

# Creamos activity_range para el modelo de clasificación    
num_data_class['activity_range'] = categorize_quantile(num_data_class, 'activity_score_pca')

# -------------------------------------------------------
# METRICS
# -------------------------------------------------------

st.divider()

# Mostramos los datos actualizados por dentro
col1, col2, col3 = st.columns(3)
col1.metric("Nuevos usuarios", f"{round(num_data['new_users'].iloc[-1])}", f"{round(num_data['new_users'].iloc[-1] - num_data['new_users'].iloc[-2])}")
col2.metric("Activity Score PCA", f"{round(num_data['activity_score_pca'].iloc[-1],2)}", f"{round(num_data['activity_score_pca'].iloc[-1] - num_data['activity_score_pca'].iloc[-2], 2)}")
#col3.metric("Impact Score", f"{round(page4.pred_score)}", f"{page4.pred_score - round(num_data['impact_score'].iloc[-1], 2)}")
col3.metric("Impact Score", f"{round(num_data['impact_score'].iloc[-1],2)}", f"{round(num_data['impact_score'].iloc[-1] - num_data['impact_score'].iloc[-2], 2)}")


# -------------------------------------------------------
# PREDICT NEW USERS & ACTIVITY
# -------------------------------------------------------

st.divider()

X = num_data.drop(columns = ['target'])
X_class = num_data_class.drop(columns = ['target'])
pred = 0
pred_class = 4

if st.button("Predict Tomorrow New Users & Activity 🎱", type='primary', use_container_width=True):
    try:
        pred = modelo.predict(X)[-1]
        pred_class = logistic_class_model.predict(X_class)[-1]
    except Exception as e:
        st.error(f"❌ Error en la predicción: {e}")

col1, col2 = st.columns(2)

with col1:
    
    st.subheader(f"Predicción de nuevos usuarios para mañana:")
    st.header(f':rainbow-background[**{round(pred)}**]')

with col2:

    st.subheader(f"Predicción de nivel de actividad para mañana:")
    pred_categories = {0:'Muy baja',1:'Baja',2:'Alta',3:'Muy alta',4:'0'}
    st.header(f':rainbow-background[{pred_categories[pred_class]}]')


