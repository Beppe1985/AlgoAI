import streamlit as st
import re
from collections import Counter
from transformers import pipeline

@st.cache_resource
def carica_modello_sentiment():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

@st.cache_resource
def carica_modello_promo():
    return pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def calcola_punteggio(post, ha_media):
    score = 0
    suggerimenti = []
    parole = post.lower().split()
    sentiment_pipeline = carica_modello_sentiment()
    promo_pipeline = carica_modello_promo()

    # [Tutte le regole base come sopra...]
    # ... (incolla le regole 1-8)

    # 9. SENTIMENT
    try:
        res = sentiment_pipeline(post[:512])[0]
        if "5 star" in res['label'] or "4 star" in res['label']:
            score += 20
        elif "3 star" in res['label']:
            score += 5
        else:
            score -= 10
    except: pass

    # 10. PROMO
    try:
        promo = promo_pipeline(post[:512])[0]
        if promo['label'] == 'negative' and promo['score'] > 0.7:
            score -= 25
    except: pass

    viralita = min(100, max(0, score))
    livello = "ALTO" if viralita > 70 else "MEDIO" if viralita > 50 else "BASSO"
    return score, viralita, livello, suggerimenti

# Interfaccia Streamlit (come prima)