# FeelText 🎭

A powerful **Multilingual** Sentiment Analysis API built with **Python**, **FastAPI**, and **Transformers** (BERT).

FeelText analyzes text input in **100+ languages** (including Persian/Farsi) and determines whether the sentiment is **positive**, **negative**, or **neutral** using state-of-the-art deep learning models.

## ✨ Features

- **🌐 Multilingual Support** - Supports 100+ languages including Persian (Farsi), Arabic, Chinese, and more
- **🎯 Accurate Analysis** - Uses multilingual DistilBERT trained on sentiment data
- **⚡ Fast & Efficient** - Optimized for quick response times
- **📊 Confidence Scores** - Get detailed confidence metrics for each prediction
- **📦 Batch Processing** - Analyze multiple texts in a single request (supports mixed languages)
- **📖 Interactive Docs** - Built-in Swagger UI and ReDoc documentation
- **🔌 REST API** - Easy integration with any application

## 🗣️ Supported Languages

| Language | Example |
|----------|---------|
| 🇺🇸 English | "I love this product!" |
| 🇮🇷 Persian (Farsi) | "این محصول عالی است!" |
| 🇸🇦 Arabic | "هذا المنتج رائع!" |
| 🇫🇷 French | "J'adore ce produit!" |
| 🇩🇪 German | "Ich liebe dieses Produkt!" |
| 🇪🇸 Spanish | "¡Me encanta este producto!" |
| 🇨🇳 Chinese | "我喜欢这个产品！" |
| 🇯🇵 Japanese | "この製品が大好きです！" |
| 🇰🇷 Korean | "이 제품이 정말 좋아요!" |
| 🇹🇷 Turkish | "Bu ürünü seviyorum!" |
| + 90 more... | |

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI |
| ML Model | Multilingual DistilBERT |
| Deep Learning | PyTorch |
| NLP | Hugging Face Transformers |
| Validation | Pydantic |

## 📋 Prerequisites

- Python 3.10+
- pip or conda

## 🚀 Quick Start

### 1. Clone the Repository

```bash
cd FeelText
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔗 API Endpoints

### Health Check
```
GET /health
```

### Supported Languages
```
GET /languages
```

### Analyze Single Text
```
POST /analyze
```

**English Example:**
```json
{
    "text": "I love this product! It's amazing."
}
```

**Persian (Farsi) Example:**
```json
{
    "text": "این محصول عالی است! خیلی راضی هستم."
}
```

**Response:**
```json
{
    "text": "این محصول عالی است! خیلی راضی هستم.",
    "sentiment": "positive",
    "confidence": 0.9534,
    "scores": {
        "positive": 0.9534,
        "negative": 0.0203,
        "neutral": 0.0263
    }
}
```

### Batch Analysis (Mixed Languages)
```
POST /analyze/batch
```

**Request Body:**
```json
{
    "texts": [
        "Great experience!",
        "این خدمات افتضاح بود",
        "C'est magnifique!",
        "Das war schrecklich"
    ]
}
```

**Response:**
```json
{
    "results": [
        {
            "text": "Great experience!",
            "sentiment": "positive",
            "confidence": 0.9876,
            "scores": {"positive": 0.9876, "negative": 0.0052, "neutral": 0.0072}
        },
        {
            "text": "این خدمات افتضاح بود",
            "sentiment": "negative",
            "confidence": 0.9234,
            "scores": {"positive": 0.0312, "negative": 0.9234, "neutral": 0.0454}
        },
        {
            "text": "C'est magnifique!",
            "sentiment": "positive",
            "confidence": 0.9654,
            "scores": {"positive": 0.9654, "negative": 0.0123, "neutral": 0.0223}
        },
        {
            "text": "Das war schrecklich",
            "sentiment": "negative",
            "confidence": 0.8932,
            "scores": {"positive": 0.0534, "negative": 0.8932, "neutral": 0.0534}
        }
    ],
    "total_analyzed": 4
}
```

## 💡 Usage Examples

### Using cURL

```bash
# English text
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is absolutely wonderful!"}'

# Persian (Farsi) text
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "این فیلم خیلی خوب بود!"}'

# Mixed language batch
curl -X POST "http://localhost:8000/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I love it!", "متنفرم از این!", "C'\''est parfait!"]}'
```

### Using Python

```python
import requests

# Persian text analysis
response = requests.post(
    "http://localhost:8000/analyze",
    json={"text": "این رستوران غذاهای خوشمزه‌ای دارد!"}
)
print(response.json())
# Output: {"text": "...", "sentiment": "positive", "confidence": 0.94, ...}

# Mixed language batch
response = requests.post(
    "http://localhost:8000/analyze/batch",
    json={"texts": [
        "Amazing product!",
        "محصول بدی بود",
        "Très bien!"
    ]}
)
print(response.json())
```

### Using JavaScript

```javascript
// Persian text analysis
const response = await fetch('http://localhost:8000/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: 'این کتاب عالی بود!' })
});
const result = await response.json();
console.log(result);
```

## 🧠 About the Model

FeelText uses **multilingual DistilBERT** (`lxyuan/distilbert-base-multilingual-cased-sentiments-student`), a model trained on multilingual sentiment data.

### Model Characteristics:
- **Architecture**: Multilingual DistilBERT
- **Languages**: 100+ languages
- **Output Classes**: Positive, Negative, Neutral
- **Base Model**: Multilingual BERT (104 languages)
- **Training**: Distilled from larger multilingual sentiment models

### Persian (Farsi) Support:
The model handles Persian text natively, including:
- Standard Persian script
- Mixed Persian-English text
- Informal/colloquial Persian
- Persian with Arabic loanwords

## 📁 Project Structure

```
FeelText/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── models.py        # Pydantic models
│   └── sentiment.py     # Multilingual sentiment service
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔬 Research Areas

This project demonstrates concepts from:

- **Natural Language Processing (NLP)**: Multilingual text processing
- **Deep Learning**: Transformer architecture, attention mechanisms
- **Transfer Learning**: Using pre-trained multilingual models
- **Cross-lingual NLP**: Processing multiple languages with one model
- **API Development**: RESTful services, request validation

## 📄 License

MIT License - feel free to use this project for learning and development.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Add support for more languages
- Submit pull requests

---

Built with ❤️ using FastAPI and Transformers | Supports 🇮🇷 Persian (Farsi) and 100+ languages
