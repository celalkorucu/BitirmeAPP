import zeyrek 
from flask import Flask, request, jsonify, render_template
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
import requests
import nltk

app = Flask(__name__)

# Whisper API bilgisi
WHISPER_API_URL = "http://13.48.129.58:5000/transcribe"
BERT_API_URL = "http://13.48.129.58:5001/classify"

# Word2Vec Model Yükleme
try:
    loaded_model = Word2Vec.load("C:/Users/90505/Desktop/BitirmeAPP/models/word_suggesiton_model/word_embeddings.model")
    print("Kelime öneri modeli başarıyla yüklendi.")
except Exception as e:
    print(f"Kelime öneri modeli yüklenirken bir hata meydana geldi: {e}")
    loaded_model = None

analyzer = zeyrek.MorphAnalyzer()

# Ana sayfa için HTML'yi render etme
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    # Dosyayı Whisper API'ye gönder
    response = requests.post(WHISPER_API_URL, files={'file': file})
    if response.status_code != 200:
        return jsonify({'error': 'Whisper API hatası'}), 500

    transcription = response.json().get('transcription', '')
    # Metni cümlelere ayır
    
    sentences = sent_tokenize(transcription)
    print(sentences)
    return jsonify({'sentences': sentences})

@app.route('/suggestions', methods=['GET'])
def get_suggestions():
    word = request.args.get('word', '').strip()
    word = word.lower()
    if not word:
        return jsonify({'error': 'Kelime parametresi eksik.'}), 400

    # Modelin yüklü olup olmadığını kontrol et
    if loaded_model is None:
        return jsonify({'error': 'Model yüklü değil.'}), 500

    try:
        # Word2Vec kullanarak önerileri al
        suggestions = [item[0] for item in loaded_model.wv.most_similar(word, topn=5)]
        print(suggestions)
        return jsonify({'suggestions': suggestions})
    except KeyError:
        # Eğer kelime modelde yoksa
        return jsonify({'suggestions': ['Bu kelime için öneri yok.']}), 200


@app.route('/bert-analyze', methods=['POST'])
def analyze_text():
    try:
        # Gelen JSON verisini al
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Gönderilen veri eksik!'}), 400

        # JS'den gelen `storedSentences` ve `wordList` yapıları
        sentences = data.get('sentences', [])
        keywords = data.get('keywords', [])
        if not sentences:
            return jsonify({'error': 'Cümleler eksik!'}), 400
        if not keywords:
            return jsonify({'error': 'Anahtar kelimeler eksik!'}), 400

        # Anahtar kelimelerin köklerini al
        keyword_roots = set()
        for keyword in keywords:
            analyzed = analyzer.analyze(keyword)
            if analyzed:  # Eğer kök analizi varsa
                keyword_roots.add(analyzed[0][0].lemma)

        # Anahtar kelimeleri içeren cümleleri bul
        matching_sentences = []
        for sentence in sentences:
            sentence_roots = []
            for word in sentence.split():
                analyzed = analyzer.analyze(word)
                if analyzed:  # Eğer kelimenin kökü varsa
                    sentence_roots.append(analyzed[0][0].lemma)

            # Anahtar kelime kökleri ile eşleşen cümleleri bul
            if any(root in keyword_roots for root in sentence_roots):
                matching_sentences.append(sentence)

        # Filtrelenen cümleler BERT modeline gönderilecek
        bert_payload = {
            "sentences": matching_sentences
        }

        bert_response = requests.post(BERT_API_URL, json=bert_payload)
        if bert_response.status_code != 200:
            return jsonify({'error': 'BERT modeline istek sırasında bir hata oluştu.'}), 500

        bert_result = bert_response.json()

        # Terminale yazdır (isteğe bağlı)
        print("Eşleşen cümleler:")
        for s in matching_sentences:
            print(s)

        print("BERT modelinden gelen sonuçlar:")
        print(bert_result)

        # Cevap dön
        return jsonify({
            'message': 'Analiz ve BERT işlemi tamamlandı!',
            'matching_sentences_count': len(matching_sentences),
            'matching_sentences': matching_sentences,  # Eşleşen cümleler
            'bert_results': bert_result  # BERT modelinden gelen sonuçlar
        })

    except Exception as e:
        return jsonify({'error': f'Sunucu tarafında bir hata oluştu: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)