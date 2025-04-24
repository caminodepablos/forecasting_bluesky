import streamlit as st

# -------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------

st.title(':gray[ABOUT ME]')
st.subheader(':gray[👋 ¡Hola! Soy Camino, gracias por interesarte por mi proyecto.]')

st.divider()


st.markdown('''
✨ Con años de experiencia en marketing digital, estrategia y gestión de producto, he podido desarrollar una visión integral del mundo digital. Siempre he estado enfocada en productos digitales, comenzando desde el marketing y evolucionando hacia el Digital Product Management.

Trabajar inmersa en el sector IT me ha permitido desarrollar habilidades que no pensé que tocaría en mi carrera profesional. Algunas más técnicas como la programación, otras más soft como las metodologías de trabajo ágiles y la gestión de equipos digitales. 

✨ Actualmente estoy inmersa en la Ciencia de Datos, habiéndome formado en Python, Machine Learning, bibliotecas especializadas como Pandas, NumPy y Seaborn, y el objetivo de dar el salto profesional a este campo.

Me encanta aprender, conectar con personas y descubrir nuevas ideas, por eso siempre estoy abierta a oportunidades y experiencias que me desafíen y me ayuden a crecer.

Si te interesa mi perfil o mi experiencia, no dudes en contactarme 😊
''')

st.divider()

# Botones 
left_letf, left, middle, right, right_right = st.columns(5, gap='small')

left_letf.markdown('**Camino de Pablos**')
left.link_button("LinkedIn :material/person:", "https://www.linkedin.com/in/caminodepablos/", use_container_width=True)
middle.link_button("GitHub :material/code:", "https://github.com/caminodepablos/", use_container_width=True)
right.link_button("Kaggle :material/raven:", "https://www.kaggle.com/caminodepablos", use_container_width=True)
right_right.link_button('Email :material/mail:', 'mailto:caminodepablos@gmail.com', type='primary', use_container_width=True)

st.divider()

