import os
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import speech_recognition as sr


def ses_ozelliklerini_cikar(dizin):
    etiketler = []
    ozellikler = []

    for dosya in os.listdir(dizin):
        if dosya.endswith('.wav'):
            dosya_yolu = os.path.join(dizin, dosya)
            y, sr = librosa.load(dosya_yolu, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs = np.mean(mfccs.T, axis=0)
            
            ozellikler.append(mfccs)
            etiketler.append(dosya.split('_')[0])  # Dosya adının ilk kısmını etiket olarak al

            # Veri artırma için yeni örnekler oluşturma
            for _ in range(3):  # Her dosya için 3 yeni örnek oluşturuyoruz
                # Zamansal kaydırma
                y_shifted = np.roll(y, np.random.randint(y.shape[0]))
                mfccs_shifted = librosa.feature.mfcc(y=y_shifted, sr=sr, n_mfcc=40)
                mfccs_shifted = np.mean(mfccs_shifted.T, axis=0)
                ozellikler.append(mfccs_shifted)
                etiketler.append(dosya.split('_')[0])
                
                # Rastgele gürültü ekleme
                y_noisy = y + 0.005 * np.random.randn(len(y))
                mfccs_noisy = librosa.feature.mfcc(y=y_noisy, sr=sr, n_mfcc=40)
                mfccs_noisy = np.mean(mfccs_noisy.T, axis=0)
                ozellikler.append(mfccs_noisy)
                etiketler.append(dosya.split('_')[0])
                
                # Rastgele hız değişikliği
                speed_change = np.random.uniform(low=0.9, high=1.1)
                y_speed = librosa.effects.time_stretch(y, rate=speed_change)
                mfccs_speed = librosa.feature.mfcc(y=y_speed, sr=sr, n_mfcc=40)
                mfccs_speed = np.mean(mfccs_speed.T, axis=0)
                ozellikler.append(mfccs_speed)
                etiketler.append(dosya.split('_')[0])
                
                # Rastgele pitç (pitch) değişikliği
                pitch_change = np.random.uniform(-5, 5)
                y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_change)
                mfccs_pitch = librosa.feature.mfcc(y=y_pitch, sr=sr, n_mfcc=40)
                mfccs_pitch = np.mean(mfccs_pitch.T, axis=0)
                ozellikler.append(mfccs_pitch)
                etiketler.append(dosya.split('_')[0])

    return np.array(ozellikler), np.array(etiketler)


# Veriyi hazırlama
veri_dizini = "sesler"  # Ses dosyalarının bulunduğu dizin
ozellikler, etiketler = ses_ozelliklerini_cikar(veri_dizini)


# Etiketleri encode etme
le = LabelEncoder()
etiketler_encoded = le.fit_transform(etiketler)

# Etiket kodlayıcısını kaydetme
np.save('label_encoder.npy', le.classes_)


# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(ozellikler, etiketler_encoded, test_size=0.2, random_state=42)


# Veriyi CNN ile çalışacak şekilde yeniden şekillendirme
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]


# CNN modelini oluşturma
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(40, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(y_train)), activation='softmax'))


# Modeli derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# Modelin performansını değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')


# Modeli kaydetme
model.save("model9.h5")


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
    

# Gerçek zamanlı ses kaydı ve tahmin fonksiyonu
def ses_kaydi_ve_tahmin(model, le, duration=5, sr=22050, threshold=0.5):
    print("Kayıt başlıyor...")
    myrecording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float64', blocking=True)  # blocking=True ile ses kaydının bitmesini bekliyoruz
    print("Kayıt tamamlandı.")
    
    # Kaydedilen sesi bir dosyaya yazma
    audio_file = "recorded_audio.wav"
    sf.write(audio_file, myrecording.flatten(), sr)
    
    # Tahmin
    y, _ = librosa.load(audio_file, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    mfccs = mfccs[np.newaxis, ..., np.newaxis]  # Modelin beklediği şekle sokma
    
    prediction = model.predict(mfccs)
    max_confidence = np.max(prediction)
    if max_confidence < threshold:
        print("Ses tanınamadı.")
    else:
        predicted_label = np.argmax(prediction, axis=1)
        predicted_speaker = le.inverse_transform(predicted_label)
        print(f"Tahmin edilen kişi: {predicted_speaker[0]} (Güven: {max_confidence:.2f})")
    
    # Ses kaydını metne dönüştürme
    text = ses_to_text(audio_file)
    print(f"Söylenen metin: {text}")
    
    # Kelime sayısını hesaplama
    word_count = len(text.split())
    print(f"Toplam kelime sayısı: {word_count}")

# Kayıt ve tahmin için modeli yükleme
model = tf.keras.models.load_model("model9.h5")


# Doğru çağrı örneği, le parametresi de eklenmiş
ses_kaydi_ve_tahmin(model, le, duration=5, sr=22050, threshold=0.5)


