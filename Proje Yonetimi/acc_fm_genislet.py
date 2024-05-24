import os
import sys
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import tensorflow as tf
import speech_recognition as sr
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline

def resource_path(relative_path):
    """PyInstaller ile uyumlu dosya yolu oluşturma"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Örnek dosya yolu kullanımı
model_path = resource_path("model9.h5")
label_encoder_path = resource_path("label_encoder.npy")

# Modeli ve etiket kodlayıcıyı yükleme
model = tf.keras.models.load_model(model_path)
le_classes = np.load(label_encoder_path, allow_pickle=True)
le = LabelEncoder()
le.classes_ = le_classes

# Duygu analizi modeli yükleme
emotion_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Ses kaydı fonksiyonu
def kaydet():
    global recorded_audio
    global sample_rate
    duration = 5
    sample_rate = 22050

    print("Kayıt başlıyor...")
    recorded_audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64', blocking=True)
    print("Kayıt tamamlandı.")

    audio_file = "recorded_audio.wav"
    sf.write(audio_file, recorded_audio.flatten(), sample_rate)

    messagebox.showinfo("Bilgi", "Ses kaydedildi.")

# Tahmin sonuçlarını göstermek için yeni pencere
def tahmin_et():
    global recorded_audio
    if recorded_audio is None:
        messagebox.showwarning("Uyarı", "Önce bir ses kaydedin.")
        return

    sample_rate = 22050
    threshold = 0.5

    audio_file = "recorded_audio.wav"
    sf.write(audio_file, recorded_audio.flatten(), sample_rate)

    y, _ = librosa.load(audio_file, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    mfccs = mfccs[np.newaxis, ..., np.newaxis]

    prediction = model.predict(mfccs)
    max_confidence = np.max(prediction)
    if max_confidence < threshold:
        result = "Ses tanınamadı."
    else:
        predicted_label = np.argmax(prediction, axis=1)
        predicted_speaker = le.inverse_transform(predicted_label)
        result = f"Tahmin edilen kişi: {predicted_speaker[0]} (Güven: {max_confidence:.2f})"

    text = ses_to_text(audio_file)
    result += f"\nSöylenen metin: {text}"

    word_count = len(text.split())
    result += f"\nToplam kelime sayısı: {word_count}"

    # ACC ve FM değerlerini hesapla
    true_label = np.array([1])  # Gerçek sınıfı burada belirtin, örneğin [1] (örnek: 1. kişi)
    acc = accuracy_score(true_label, predicted_label)
    fm = f1_score(true_label, predicted_label, average='binary')
    result += f"\nACC Değeri: {acc:.2f}\nFM Değeri: {fm:.2f}"

    # Yeni pencere oluşturma ve tahmin sonuçlarını gösterme
    result_window = tk.Toplevel(root)
    result_window.title("Tahmin Sonuçları")

    result_label = tk.Label(result_window, text=result, font=("Helvetica", 14))
    result_label.pack(padx=20, pady=20)

# Histogram oluşturma fonksiyonu
def histogram_olustur():
    global recorded_audio
    if recorded_audio is None:
        messagebox.showwarning("Uyarı", "Önce bir ses kaydedin.")
        return

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(recorded_audio)) / sample_rate, recorded_audio)
    plt.xlabel('Zaman (s)')
    plt.ylabel('Amplitüd')

    plt.subplot(2, 1, 2)
    plt.specgram(recorded_audio.flatten(), Fs=sample_rate)
    plt.xlabel('Zaman (s)')
    plt.ylabel('Frekans (Hz)')
    plt.colorbar(label='Güç (dB)')

    plt.tight_layout()
    plt.show()

    # Sesin enerjisi ve zaman uzunluğunu yazdır
    enerji = np.sum(np.abs(recorded_audio)**2)
    zaman_uzunlugu = len(recorded_audio) / sample_rate
    print("Ses Enerjisi:", enerji)
    print("Zaman Uzunluğu:", zaman_uzunlugu)

# Duygu tahmini işlevi
def duygu_tahmini():
    global recorded_audio
    if recorded_audio is None:
        messagebox.showwarning("Uyarı", "Önce bir ses kaydedin.")
        return

    audio_file = "recorded_audio.wav"
    text = ses_to_text(audio_file)

    # Duygu analizi
    emotions = emotion_analyzer(text)
    result = f"Söylenen metin: {text}\n\nDuygu Tahmini:\n"
    positive_score = emotions[0]['score']
    negative_score = 1 - positive_score
    result += f"    Pozitif: %{positive_score * 100:.2f}\n"
    result += f"    Negatif: %{negative_score * 100:.2f}\n"

    # Yeni pencere oluşturma ve duygu analiz sonuçlarını gösterme
    emotion_window = tk.Toplevel(root)
    emotion_window.title("Duygu Analizi Sonuçları")

    emotion_label = tk.Label(emotion_window, text=result, font=("Helvetica", 14))
    emotion_label.pack(padx=20, pady=20)

# SpeechRecognition kullanarak ses kaydını metne dönüştürme fonksiyonu
def ses_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language="tr-TR")
        return text
    except sr.UnknownValueError:
        return "Ses anlaşılamadı"
    except sr.RequestError as e:
        return f"Google API hatası; {e}"

# Tkinter arayüzü
root = tk.Tk()
root.title("Ses Tanıma Uygulaması")

# Başlangıçta kaydedilen ses boş olacak
recorded_audio = None
sample_rate = 22050

# Arayüz öğelerinin düzenlenmesi
title_label = tk.Label(root, text="Ses Tanıma Uygulaması", font=("Helvetica", 36, "bold"), fg="black")
title_label.pack(pady=20)

button_font = ("Helvetica", 24, "bold")
button_width = 20  # Buton genişliği

kaydet_button = tk.Button(root, text="Kaydet", command=kaydet, bg="green", fg="white", font=button_font, width=button_width)
kaydet_button.pack(pady=10)

tahmin_et_button = tk.Button(root, text="Tahmin Et", command=tahmin_et, bg="orange", fg="white", font=button_font, width=button_width)
tahmin_et_button.pack(pady=10)

histogram_button = tk.Button(root, text="Histogram Oluştur", command=histogram_olustur, bg="#E6E6FA", fg="black", font=button_font, width=button_width)
histogram_button.pack(pady=10)

duygu_button = tk.Button(root, text="Duygu Tahmini", command=duygu_tahmini, bg="#DDA0DD", fg="white", font=button_font, width=button_width)
duygu_button.pack(pady=10)

root.mainloop()
