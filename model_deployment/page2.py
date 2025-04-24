import streamlit as st

# -------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------

st.title(':gray[BLUESKY]')
st.subheader(':gray[*Welcome to the social internet.*]')

# Botones para visitar Bluesky
left, middle, right = st.columns(3, gap='small')
left.link_button("Visit Bluesky App 👉", "https://bsky.app/", use_container_width=True)
middle.link_button("Visit Bluesky About Web 🦋", "https://bsky.social/about", use_container_width=True)

st.divider()


# Comic
st.image("https://bsky.social/about/welcome-to-bluesky-comic-davis-bickford/cover.jpg")
st.image("https://bsky.social/about/welcome-to-bluesky-comic-davis-bickford/page-1.jpg")
st.image("https://bsky.social/about/welcome-to-bluesky-comic-davis-bickford/page-2.jpg")
st.image("https://bsky.social/about/welcome-to-bluesky-comic-davis-bickford/page-3.jpg")
st.caption('Comic by @davis.social and Bluesky. [See original source](https://bsky.social/about/welcome-to-bluesky-comic)')



