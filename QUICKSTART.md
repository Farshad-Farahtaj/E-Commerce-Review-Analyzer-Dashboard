# 🔐 Secure Quick Start Guide

## 🚀 Get Started in 5 Minutes with Full Authentication

### Option 1: Run Locally with Security

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Launch secure dashboard**
```bash
streamlit run app.py
```

3. **Register/Login**
   - Navigate to `http://localhost:8501`
   - **First time**: Click "Register" to create account
   - **Returning**: Login with your credentials
   - **Security**: All passwords encrypted, sessions tracked

4. **Start analyzing securely**
   - Access personalized user dashboard
   - View your activity history
   - Analyze reviews with full audit trail

### Option 2: Run with Docker

1. **Build and run**
```bash
docker build -t review-analyzer .
docker run -p 8501:8501 review-analyzer
```

2. **Access dashboard**
   - Open `http://localhost:8501`

### Option 3: Docker Compose

```bash
docker-compose up -d
```

## 📊 Test the System

### Run Accuracy Assessment
```bash
python test_accuracy.py
```

This will:
- Evaluate model on 100 labeled reviews
- Generate accuracy report
- Create visualizations

Expected output: **~94% accuracy**

## 🎯 Using the Dashboard

### Mode 1: Analyze Sample Dataset
1. Select "📁 Analyze Sample Dataset"
2. View metrics and charts
3. Read AI-generated insights

### Mode 2: Analyze Custom Reviews
1. Select "✍️ Analyze Custom Reviews"
2. Paste reviews (one per line)
3. Click "Analyze"
4. View results instantly

## 🐛 Troubleshooting

**Issue: Models loading slowly**
- First load takes 30-60 seconds (models are ~500MB)
- Subsequent loads are cached

**Issue: Out of memory**
- Reduce batch size in code
- Use smaller model variants
- Increase Docker memory limit

**Issue: processed_reviews.csv not found**
- Either run Phase 1 notebook first
- Or use "Analyze Custom Reviews" mode

## 📚 Next Steps

1. ✅ Test locally
2. ✅ Run accuracy assessment
3. ✅ Deploy with Docker
4. ✅ Push to GitHub
5. ✅ Deploy to cloud (Streamlit Cloud, AWS, etc.)

## 💡 Tips

- Start with "Custom Reviews" mode if you don't have processed data
- Use Google Colab for Phase 1 (free GPU access)
- Docker deployment is production-ready
- Check README.md for detailed documentation
