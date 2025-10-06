import nltk
import re
from flask import Flask, render_template, request, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import langdetect
from werkzeug.utils import secure_filename
import os

# -----------------------
# Flask app initialization
# -----------------------
app = Flask(__name__, static_url_path='/static')

# Use Vercel's writable temp directory
UPLOAD_FOLDER = '/tmp/temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# -----------------------
# Sentiment and Translation Setup
# -----------------------
analyzer = SentimentIntensityAnalyzer()
TRANSLATION_ENABLED = False

try:
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source='auto', target='en')
    TRANSLATION_ENABLED = True
except ImportError:
    translator = None

# -----------------------
# Unified emotion keywords (English-based)
# -----------------------
EMOTION_KEYWORDS = {
    'joy': ['happy', 'excited', 'delighted', 'cheerful', 'elated', 'thrilled', 'overjoyed', 'jubilant', 'ecstatic', 'blissful', 'content', 'pleased', 'glad', 'amazing', 'wonderful', 'fantastic', 'great', 'awesome', 'brilliant', 'excellent'],
    'sadness': ['sad', 'depressed', 'melancholy', 'sorrowful', 'mournful', 'dejected', 'despondent', 'heartbroken', 'grief', 'disappointed', 'upset', 'down', 'blue', 'gloomy', 'miserable', 'unhappy', 'terrible', 'awful', 'horrible'],
    'anger': ['angry', 'furious', 'enraged', 'livid', 'outraged', 'irritated', 'annoyed', 'frustrated', 'mad', 'hostile', 'aggressive', 'rage', 'hate', 'disgusted', 'infuriated', 'pissed', 'damn', 'stupid', 'ridiculous'],
    'fear': ['afraid', 'scared', 'terrified', 'frightened', 'anxious', 'worried', 'nervous', 'panic', 'alarmed', 'apprehensive', 'concerned', 'uneasy', 'stressed', 'overwhelmed', 'insecure', 'uncertain', 'doubtful'],
    'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'bewildered', 'confused', 'unexpected', 'sudden', 'wow', 'incredible', 'unbelievable', 'remarkable', 'extraordinary'],
    'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'nauseated', 'appalled', 'horrified', 'gross', 'yuck', 'eww', 'nasty', 'foul', 'repugnant'],
    'trust': ['trust', 'confident', 'reliable', 'secure', 'safe', 'comfortable', 'assured', 'certain', 'dependable', 'faithful', 'loyal', 'honest', 'sincere'],
    'anticipation': ['excited', 'eager', 'looking forward', 'anticipating', 'expecting', 'hopeful', 'optimistic', 'ready', 'prepared', 'waiting', "can't wait"]
}

# -----------------------
# Emotion emojis
# -----------------------
EMOTION_EMOJIS = {
    'joy': '😄',
    'sadness': '😢',
    'anger': '😡',
    'fear': '😨',
    'surprise': '😲',
    'disgust': '🤢',
    'trust': '🤝',
    'anticipation': '🤗',
    'neutral': '😐'
}

# -----------------------
# Language mapping
# -----------------------
LANGUAGE_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
    'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi'
}

# -----------------------
# Helper functions
# -----------------------
def detect_language(text):
    try:
        return langdetect.detect(text)
    except:
        return 'en'

def translate_text(text, source_lang='auto', target_lang='en'):
    if not TRANSLATION_ENABLED or source_lang == target_lang or source_lang == 'en':
        return text
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def get_emotion_from_text(text, sentiment_scores):
    text_lower = text.lower()
    emotion_scores = {k: sum(1 for kw in v if kw in text_lower) for k, v in EMOTION_KEYWORDS.items()}

    if any(emotion_scores.values()):
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
    else:
        compound = sentiment_scores['compound']
        if compound >= 0.5:
            primary_emotion = 'joy'
        elif compound <= -0.5:
            primary_emotion = 'sadness'
        elif sentiment_scores.get('neu', 0) > 0.7:
            primary_emotion = 'neutral'
        else:
            primary_emotion = 'anticipation' if sentiment_scores.get('pos',0) > sentiment_scores.get('neg',0) else 'fear'
    return primary_emotion, emotion_scores

def analyze_sentiment_consistent(text, original_language='en'):
    try:
        english_text = translate_text(text, original_language, 'en') if original_language != 'en' else text
        sentiment_scores = analyzer.polarity_scores(english_text)
        compound = sentiment_scores['compound']
        sentiment = 'Positive' if compound >= 0.05 else 'Negative' if compound <= -0.05 else 'Neutral'
        return sentiment, sentiment_scores, english_text
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        sentiment_scores = analyzer.polarity_scores(text)
        compound = sentiment_scores['compound']
        sentiment = 'Positive' if compound >= 0.05 else 'Negative' if compound <= -0.05 else 'Neutral'
        return sentiment, sentiment_scores, text

# -----------------------
# Routes
# -----------------------
@app.route('/')
def index():
    return render_template('in.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text','').strip()
        if not text:
            return jsonify({'error':'No text provided'}),400

        language = detect_language(text)
        language_name = LANGUAGE_NAMES.get(language, language.upper())
        sentiment, sentiment_scores, analyzed_text = analyze_sentiment_consistent(text, language)
        primary_emotion, emotion_scores = get_emotion_from_text(analyzed_text, sentiment_scores)
        emoji = EMOTION_EMOJIS.get(primary_emotion,'😐')
        confidence = abs(sentiment_scores['compound'])

        return jsonify({
            'sentiment': sentiment,
            'emotion': primary_emotion,
            'emoji': emoji,
            'confidence': round(confidence,3),
            'language': language,
            'language_name': language_name,
            'original_text': text,
            'analyzed_text': analyzed_text if analyzed_text != text else None,
            'detailed_scores': {
                'positive': round(sentiment_scores['pos'],3),
                'negative': round(sentiment_scores['neg'],3),
                'neutral': round(sentiment_scores['neu'],3),
                'compound': round(sentiment_scores['compound'],3)
            },
            'emotion_breakdown': emotion_scores,
            'analysis_method':'consistent_english_based',
            'translation_used': analyzed_text != text
        })
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}),500

@app.route('/analyze-batch', methods=['POST'])
def analyze_batch():
    try:
        texts = []

        # File upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error':'No file selected'}),400
            if file.filename.lower().endswith('.txt'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                with open(filepath,'r',encoding='utf-8',errors='ignore') as f:
                    texts = [line.strip() for line in f.readlines() if line.strip()]
                os.remove(filepath)

        else:
            data = request.get_json()
            texts = data.get('texts',[])

        if not texts:
            return jsonify({'error':'No texts provided for analysis'}),400

        results = []
        language_counts = {}

        for i,text in enumerate(texts[:50]):
            if not text.strip():
                continue
            language = detect_language(text)
            language_name = LANGUAGE_NAMES.get(language, language.upper())
            language_counts[language_name] = language_counts.get(language_name,0)+1
            sentiment, sentiment_scores, analyzed_text = analyze_sentiment_consistent(text,language)
            primary_emotion, emotion_scores = get_emotion_from_text(analyzed_text,sentiment_scores)
            emoji = EMOTION_EMOJIS.get(primary_emotion,'😐')
            confidence = abs(sentiment_scores['compound'])
            results.append({
                'id':i+1,
                'text':text[:100]+'...' if len(text)>100 else text,
                'sentiment':sentiment,
                'emotion':primary_emotion,
                'emoji':emoji,
                'confidence':round(confidence,3),
                'language':language,
                'language_name':language_name,
                'compound_score':round(sentiment_scores['compound'],3),
                'translation_used':analyzed_text!=text
            })

        successful_results = [r for r in results if 'error' not in r]
        if successful_results:
            avg_sentiment = sum(r['compound_score'] for r in successful_results)/len(successful_results)
            sentiment_distribution = {}
            emotion_distribution = {}
            for r in successful_results:
                sentiment_distribution[r['sentiment']] = sentiment_distribution.get(r['sentiment'],0)+1
                emotion_distribution[r['emotion']] = emotion_distribution.get(r['emotion'],0)+1
            summary = {
                'total_analyzed':len(successful_results),
                'average_sentiment_score':round(avg_sentiment,3),
                'sentiment_distribution':sentiment_distribution,
                'emotion_distribution':emotion_distribution,
                'language_distribution':language_counts,
                'most_common_emotion':max(emotion_distribution,key=emotion_distribution.get) if emotion_distribution else 'neutral',
                'analysis_method':'consistent_english_based'
            }
        else:
            summary={'total_analyzed':0,'error':'No texts could be analyzed'}

        return jsonify({'results':results,'summary':summary})
    except Exception as e:
        return jsonify({'error': f'Batch analysis failed: {str(e)}'}),500

@app.route('/compare-analysis', methods=['POST'])
def compare_analysis():
    try:
        data = request.get_json()
        texts = data.get('texts',[])
        if not texts or len(texts)<2:
            return jsonify({'error':'Need at least 2 texts for comparison'}),400

        comparison_results=[]
        for text in texts:
            if not text.strip(): continue
            language = detect_language(text)
            language_name = LANGUAGE_NAMES.get(language,language.upper())
            sentiment,scores,analyzed_text = analyze_sentiment_consistent(text,language)
            primary_emotion, emotion_scores = get_emotion_from_text(analyzed_text,scores)
            comparison_results.append({
                'original_text':text,
                'language':language_name,
                'analyzed_text':analyzed_text,
                'sentiment':sentiment,
                'emotion':primary_emotion,
                'compound_score':round(scores['compound'],3),
                'confidence':round(abs(scores['compound']),3),
                'translation_used':analyzed_text!=text
            })

        sentiments = [r['sentiment'] for r in comparison_results]
        emotions = [r['emotion'] for r in comparison_results]
        compound_scores = [r['compound_score'] for r in comparison_results]

        consistency_check={
            'all_same_sentiment':len(set(sentiments))==1,
            'all_same_emotion':len(set(emotions))==1,
            'score_variance':round(max(compound_scores)-min(compound_scores),3),
            'is_consistent':len(set(sentiments))==1 and len(set(emotions))==1
        }

        return jsonify({'comparison_results':comparison_results,'consistency_check':consistency_check})

    except Exception as e:
        return jsonify({'error': f'Comparison failed: {str(e)}'}),500

# -----------------------
# Run Flask
# -----------------------
if __name__ == "__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
