# Fake News Detection System

## Overview
This project is a Fake News Detection System that predicts whether a news headline is real or fake using a BERT model and web verification via the Google Custom Search API. Built with Streamlit, it features a modern, user-friendly interface with custom styling. Focused on Indian news (e.g., 2025 online gaming ban), it combats misinformation with a robust pipeline.

## Features
- **BERT Model**: Trained on a full dataset (~44,266 samples) with 3 epochs, `max_length=128`, achieving ~93% accuracy.
- **Web Verification**: Uses Google Custom Search API (100 free queries/day) to validate news against credible sources (e.g., BBC, NDTV, Reuters).
- **Streamlit UI**: Polished interface with result cards, example headlines, and a warning for BERT overconfidence on fake news.
- **Evaluation**: Metrics and confusion matrices for BERT, Logistic Regression, Random Forest, and Naive Bayes models.
