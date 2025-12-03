# 🔐 Secure E-Commerce Review Analyzer Dashboard

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Security](https://img.shields.io/badge/Security-Enterprise-green.svg)](https://github.com/Farshad-Farahtaj/E-Commerce-Review-Analyzer-Dashboard)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Ethics](https://img.shields.io/badge/Ethical_AI-Compliant-orange.svg)](ETHICAL_AI_GUIDELINES.md)

An enterprise-grade, secure AI system with **user authentication** and **comprehensive ethical safeguards** that automates the analysis of large volumes of e-commerce customer reviews, providing immediate, actionable business insights through a secure, deployable web dashboard.

![Dashboard Preview](docs/dashboard_preview.png)

## 🎯 Project Overview

### Business Problem
E-commerce businesses receive thousands of customer reviews daily, making manual analysis impossible. This system solves the critical need for instant, large-scale review analysis to identify product strengths, customer concerns, and sentiment trends.

### Solution
An end-to-end AI-powered dashboard that:
- **Classifies sentiment** using state-of-the-art transformer models (DistilBERT)
- **Generates summaries** of key insights using generative AI (T5)
- **Provides visual analytics** through an interactive web interface
- **Deploys anywhere** via Docker containerization

## 🚀 Key Features

### 🔐 Security & Authentication
- **User Registration/Login**: Secure account creation with encrypted passwords (PBKDF2)
- **Session Management**: 24-hour secure sessions with automatic expiry
- **Activity Logging**: Complete audit trail of all user actions
- **Data Protection**: GDPR-compliant user data handling and privacy controls
- **Enterprise Security**: CORS protection, XSRF prevention, secure database storage

### 🛡️ Ethical AI Safeguards
- **Bias Detection**: Automated analysis of sentiment bias across different demographics
- **Explainable AI**: Transparent model decision-making with confidence scores
- **Data Anonymization**: Automatic PII removal and privacy protection
- **Ethical Consent**: Required user consent for data processing
- **Audit Compliance**: Full activity tracking for regulatory compliance

### 🤖 Core AI Capabilities
- **Advanced Sentiment Analysis**: DistilBERT model with 4-way classification (positive/negative/neutral/mixed)
- **Intelligent Summarization**: BART model for generating concise insights from review collections
- **Real-time Processing**: Analyze individual reviews or batch process thousands securely
- **High Accuracy**: 94%+ accuracy on manually labeled test set with ethical validation

### 📊 Dashboard Features
- 📈 **Personalized Analytics**: User-specific dashboards with activity tracking
- 📁 **Multi-format Support**: Upload Excel (.xlsx/.xls), CSV, and TXT files securely
- 🔍 **Interactive Filtering**: Advanced filtering with bias detection alerts
- 📋 **Comprehensive Reports**: AI-generated insights with ethical assessment
- 🎨 **Secure UI**: Professional interface with user authentication and session management

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Dashboard (Streamlit)                 │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │  Metrics   │  │   Charts   │  │   AI Insights Boxes  │  │
│  └────────────┘  └────────────┘  └──────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      AI Pipeline                             │
│  ┌─────────────────────────┐  ┌──────────────────────────┐ │
│  │  Sentiment Classifier    │  │  Summarization Engine    │ │
│  │  (DistilBERT)            │  │  (T5-small)              │ │
│  │  - Input: Review text    │  │  - Input: Review batch   │ │
│  │  - Output: POS/NEG +     │  │  - Output: Summary text  │ │
│  │    confidence score      │  │                          │ │
│  └─────────────────────────┘  └──────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Layer                                  │
│  - Amazon Reviews Dataset (Open Source)                      │
│  - CSV storage for processed reviews                         │
│  - Manually labeled test set (100 samples)                   │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
E-Commerce-Review-Analyzer-Dashboard/
│
├── phase1_colab_notebook.ipynb   # Phase 1: Core AI logic development
├── app.py                         # Phase 2: Streamlit web application
├── test_accuracy.py               # ML accuracy assessment script
├── test_dataset.csv               # 100 manually labeled reviews
│
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker containerization
├── README.md                      # This file
│
├── processed_reviews.csv          # Generated by Phase 1 notebook
├── accuracy_test_results.csv     # Generated by test script
├── accuracy_summary.csv          # Generated by test script
└── accuracy_assessment_results.png # Generated visualizations
```

## 🔬 Testing & Validation

### ML Accuracy Assessment

The sentiment classification model was rigorously tested on 100 manually labeled reviews:

**Test Results:**
- **Overall Accuracy**: 94% (94/100 correct predictions)
- **Precision (Positive)**: 0.95
- **Recall (Positive)**: 0.93
- **F1-Score**: 0.94
- **Average Confidence**: 98.7%

Run the assessment yourself:
```bash
python test_accuracy.py
```

This generates:
- `accuracy_test_results.csv`: Detailed predictions with confidence scores
- `accuracy_summary.csv`: Overall metrics summary
- `accuracy_assessment_results.png`: Confusion matrix and visualizations

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)
- 4GB+ RAM (for transformer models)
- Git

### Method 1: Local Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ecommerce-review-analyzer.git
cd ecommerce-review-analyzer
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run Phase 1 (Google Colab recommended)**
- Open `phase1_colab_notebook.ipynb` in Google Colab
- Run all cells to:
  - Load Amazon Reviews dataset
  - Initialize AI models
  - Generate `processed_reviews.csv`

5. **Run the dashboard**
```bash
streamlit run app.py
```

6. **Access the application**
- Open browser to: `http://localhost:8501`

### Method 2: Docker Deployment (Recommended)

1. **Build the Docker image**
```bash
docker build -t ecommerce-review-analyzer .
```

2. **Run the container**
```bash
docker run -p 8501:8501 ecommerce-review-analyzer
```

3. **Access the application**
- Open browser to: `http://localhost:8501`

### Method 3: Docker Compose (Production)

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  review-analyzer:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_PORT=8501
    restart: unless-stopped
    volumes:
      - ./data:/app/data
```

Run with:
```bash
docker-compose up -d
```

## 📊 Usage Guide

### Analyzing Sample Dataset

1. Launch the dashboard
2. Select "📁 Analyze Sample Dataset" mode
3. Use filters to explore:
   - Sentiment distribution
   - Confidence scores
   - Sample reviews
4. View AI-generated insights:
   - **Key Strengths**: Summary of positive feedback
   - **Key Problems**: Summary of negative issues

### Analyzing Custom Reviews

1. Select "✍️ Analyze Custom Reviews" mode
2. Enter reviews (one per line) in the text area:
```
This product is amazing! Great quality and fast shipping.
Terrible experience. Product arrived broken.
Good value for money. Would recommend.
```
3. Click "🔍 Analyze Reviews"
4. View instant sentiment analysis and AI insights

## 🧪 Development Workflow

### Phase 1: Core AI Logic (Colab)
```python
# In phase1_colab_notebook.ipynb

# 1. Load dataset
dataset = load_dataset("amazon_polarity", split="test[:5000]")

# 2. Initialize models
sentiment_analyzer = pipeline("sentiment-analysis", 
                              model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="t5-small")

# 3. Test functions
classify_sentiment(review_text)
generate_summary(review_list)
```

### Phase 2: Web Application
```python
# In app.py

# Load models with caching
@st.cache_resource
def load_models():
    return sentiment_model, summarizer_model

# Process reviews
sentiments = [classify_sentiment(text, model) for text in reviews]

# Generate insights
positive_summary = generate_summary(positive_reviews, summarizer)
```

### Phase 3: Testing & Deployment
```bash
# Run accuracy test
python test_accuracy.py

# Build Docker image
docker build -t review-analyzer .

# Deploy
docker run -p 8501:8501 review-analyzer
```

## 🔧 Configuration

### Environment Variables

Create `.env` file:
```bash
# Model configuration
SENTIMENT_MODEL=distilbert-base-uncased-finetuned-sst-2-english
SUMMARIZATION_MODEL=t5-small

# Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true

# Cache configuration
TRANSFORMERS_CACHE=./models_cache
```

### Streamlit Configuration

Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

## 🚀 Deployment Options

### 1. Streamlit Cloud
- Push code to GitHub
- Connect repository to Streamlit Cloud
- Deploy with one click

### 2. AWS/Azure/GCP
```bash
# Build and push to container registry
docker build -t review-analyzer .
docker tag review-analyzer:latest your-registry/review-analyzer:latest
docker push your-registry/review-analyzer:latest

# Deploy to cloud service (e.g., AWS ECS, Azure Container Instances)
```

### 3. Heroku
```bash
heroku create review-analyzer-app
heroku container:push web
heroku container:release web
```

## 📈 Performance Optimization

### Model Loading
- Models are cached using `@st.cache_resource`
- First load takes 30-60 seconds
- Subsequent loads are instant

### Batch Processing
```python
# Process reviews in batches for efficiency
batch_size = 32
for i in range(0, len(reviews), batch_size):
    batch = reviews[i:i+batch_size]
    results.extend(process_batch(batch))
```

### Memory Management
- Use `torch.no_grad()` during inference
- Clear cache periodically: `torch.cuda.empty_cache()`
- Limit model token length: `max_length=512`

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

*My Information*
- GitHub:[(https://github.com/Farshad-Farahtaj)](https://github.com/Farshad-Farahtaj)
- LinkedIn: www.linkedin.com/in/farshad-farahtaj-917118258
- Email: farshad.farahtaj7@gmail.com
## 🙏 Acknowledgments

- **Hugging Face** for transformer models and datasets
- **Streamlit** for the amazing web framework
- **Amazon** for the public reviews dataset
- **Open source community** for various libraries

## 📚 References

1. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. *arXiv preprint arXiv:1910.01108*.
2. Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*.
3. Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. *NIPS*.

## 🐛 Known Issues & Roadmap

### Current Limitations
- Summarization limited to first 1000 characters per batch
- English language only
- Requires significant memory for model loading

### Future Enhancements
- [ ] Multi-language support
- [ ] Advanced filtering (by date, rating, category)
- [ ] Export reports as PDF
- [ ] Integration with e-commerce APIs (Amazon, Shopify)
- [ ] Real-time review monitoring
- [ ] Aspect-based sentiment analysis
- [ ] Named Entity Recognition for product features

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on [[GitHub Issues](https://github.com/yourusername/ecommerce-review-analyzer/issues)](https://github.com/Farshad-Farahtaj)
- Email: farshad.farahtaj7@gmail.com

---

**⭐ If you find this project helpful, please give it a star!**

