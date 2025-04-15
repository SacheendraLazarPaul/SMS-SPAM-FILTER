# utils.py
from wordfreq import word_frequency
import re

def is_gibberish(text):
    words = text.split()
    if not words:
        return True
    real_words = sum(1 for word in words if word_frequency(word.lower(), 'en') > 1e-6)
    return real_words / len(words) < 0.4

def detect_alert(message):
    alert_keywords = ['account', 'bank', 'verify', 'password', 'alert', 'update', 'important', 'security']
    return any(word in message.lower() for word in alert_keywords)
