
# main.py
import streamlit as st
import re
from collections import Counter
from transformers import pipeline

# === CARICA MODELLO AI LEGGERO (veloce su Replit) ===
@st.cache_resource
def carica_modello_sentiment():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

# === FUNZIONE ANALISI ===
def calcola_punteggio(post, ha_media):
    score = 0
    suggerimenti = []
    parole = post.lower().split()

    # 1. Lunghezza
    if 100 <= len(parole) <= 500:
        score += 20
    elif len(parole) < 100:
        suggerimenti.append("Aggiungi più dettagli (min. 100 parole).")
    else:
        suggerimenti.append("Accorcia il post (max 500 parole).")

    # 2. Domanda
    if "?" in post:
        score += 15
    else:
        suggerimenti.append("Aggiungi una domanda: 'Che ne pensate?'")

    # 3. Hashtag
    hashtags = re.findall(r"#\w+", post)
    if 1 <= len(hashtags) <= 3:
        score += 10
    elif len(hashtags) > 3:
        suggerimenti.append(f"Troppi hashtag ({len(hashtags)}). Max 3.")

    # 4. Link esterni
    if "http" in post.lower() or "www." in post.lower():
        score -= 30
        suggerimenti.append("Rimuovi il link! Mettilo nei commenti.")

    # 5. Invito al dialogo
    parole_dialogo = ["pensi", "tu", "esperienza", "commenta", "secondo te"]
    if any(p in post.lower() for p in parole_dialogo):
        score += 20

    # 6. Parole di valore
    parole_valore = ["lezione", "insight", "consiglio", "errore", "strategia"]
    if sum(p in parole for p in parole_valore) >= 2:
        score += 15

    # 7. Media
    if ha_media:
        score += 25
    else:
        suggerimenti.append("Aggiungi **carousel** o **video nativo**!")

    # 8. Engagement bait
    bait = ["like se", "commenta sì", "tagga un amico"]
    if any(frase in post.lower() for frase in bait):
        score -= 20
        suggerimenti.append("Evita 'engagement bait'!")

    # 9. Sentiment AI
    try:
        result = carica_modello_sentiment()(post[:512])[0]
        label = result['label'].upper()
        if label in ["POSITIVE", "5_STARS"]:
            score += 20
            suggerimenti.append(f"Sentimento **positivo** ({result['score']:.0%})")
        elif label == "NEUTRAL":
            score += 5
        else:
            score -= 10
            suggerimenti.append(f"Sentimento **negativo** – evita toni critici")
    except:
        suggerimenti.append("AI non disponibile (errore temporaneo)")

    viralita = min(100, max(0, score))
    livello = "ALTO" if viralita > 70 else "MEDIO" if viralita > 50 else "BASSO"
    return score, viralita, livello, suggerimenti

# === INTERFACCIA ===
st.set_page_config(page_title="LinkedIn Viral Score", page_icon="Chart Increasing")
st.title("LinkedIn Post Analyzer 2025")
st.markdown("**Testa se il tuo post diventerà virale – con AI!**")

post = st.text_area("Incolla il tuo post:", height=200)
ha_media = st.checkbox("Include **carousel/video nativo**?")

if st.button("Analizza!", type="primary"):
    if post.strip():
        with st.spinner("Analisi in corso..."):
            p, v, l, s = calcola_punteggio(post, ha_media)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Punteggio", f"{p}/100")
        with col2:
            st.metric("Viralità", f"{v}%", delta=l)
        if s:
            st.subheader("Suggerimenti:")
            for sg in s:
                st.write(f"• {sg}")
        else:
            st.balloons()
            st.success("Perfetto! Post ottimizzato")
    else:
        st.error("Scrivi un post!")

st.caption("Creato per LinkedIn | AI leggera")
