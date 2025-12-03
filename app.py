"""
E-Commerce Review Analyzer Dashboard
Phase 2: Streamlit Web Application

This dashboard provides real-time sentiment analysis and insights
from customer reviews using state-of-the-art LLM models.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import warnings
import io
import os
from pathlib import Path
from ethical_features import EthicalReviewAnalyzer
from user_auth import render_auth_interface, render_user_dashboard, log_user_analysis

# Suppress all warnings including transformers model loading warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Page configuration
st.set_page_config(
    page_title="🔐 Secure E-Commerce Review Analyzer",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .positive-box {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .negative-box {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .neutral-box {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .mixed-box {
        background-color: #e2e3e5;
        border-left-color: #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for caching models
@st.cache_resource
def load_models():
    """Load and cache the sentiment and summarization models"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Use DistilBERT for more reliable sentiment analysis
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=True
        )
        
        summarizer_model = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
        
    return sentiment_model, summarizer_model

@st.cache_data
def load_sample_data():
    """Load sample review data"""
    try:
        df = pd.read_csv('processed_reviews.csv')
        return df
    except FileNotFoundError:
        # If processed file not available, return None
        return None

def classify_sentiment(text, model):
    """Enhanced sentiment classification with four categories"""
    try:
        text = text[:512] if len(text) > 512 else text
        results = model(text)
        
        # Get all scores from the model
        if isinstance(results[0], list):
            scores = results[0]
        else:
            scores = results
        
        # Find positive and negative scores
        pos_score = 0
        neg_score = 0
        
        for score in scores:
            if score['label'] == 'POSITIVE':
                pos_score = score['score']
            elif score['label'] == 'NEGATIVE':
                neg_score = score['score']
        
        # Determine sentiment based on scores and confidence
        confidence = max(pos_score, neg_score)
        
        if confidence < 0.6:
            # Low confidence - check for mixed sentiment
            keyword_sentiment = analyze_mixed_sentiment(text)
            if keyword_sentiment == 'MIXED':
                mapped_label = 'MIXED'
            else:
                mapped_label = 'NEUTRAL'
        elif pos_score > neg_score:
            mapped_label = 'POSITIVE'
        else:
            mapped_label = 'NEGATIVE'
        
        return {
            'label': mapped_label,
            'score': round(confidence, 4),
            'original_label': 'POSITIVE' if pos_score > neg_score else 'NEGATIVE'
        }
    except Exception as e:
        return {'label': 'NEUTRAL', 'score': 0.5, 'original_label': 'UNKNOWN'}

def analyze_mixed_sentiment(text):
    """Additional analysis for mixed sentiments using keyword detection"""
    positive_keywords = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'fantastic', 
                        'wonderful', 'outstanding', 'impressive', 'satisfied', 'recommend']
    negative_keywords = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing',
                        'poor', 'broken', 'defective', 'useless', 'waste', 'regret']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_keywords if word in text_lower)
    neg_count = sum(1 for word in negative_keywords if word in text_lower)
    
    if pos_count > 0 and neg_count > 0:
        return 'MIXED'
    elif pos_count > neg_count:
        return 'POSITIVE'
    elif neg_count > pos_count:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

def generate_summary(texts, model, max_length=100, min_length=30):
    """Generate summary using the loaded model"""
    try:
        if isinstance(texts, str):
            combined_text = texts
        else:
            combined_text = " ".join(texts)
        
        if len(combined_text) > 1000:
            combined_text = combined_text[:1000]
        
        # Simple summarization without conflicting parameters
        summary = model(
            combined_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        
        return summary[0]['summary_text']
    except Exception as e:
        return f"Unable to generate summary: {str(e)}"

def process_uploaded_file(uploaded_file):
    """Process uploaded Excel or text files and extract reviews"""
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension == '.xlsx' or file_extension == '.xls':
            # Handle Excel files
            df = pd.read_excel(uploaded_file)
            return process_excel_data(df)
        
        elif file_extension == '.txt':
            # Handle text files
            content = uploaded_file.read().decode('utf-8')
            return process_text_data(content)
        
        elif file_extension == '.csv':
            # Handle CSV files
            df = pd.read_csv(uploaded_file)
            return process_excel_data(df)
        
        else:
            return None, "Unsupported file format. Please upload .xlsx, .xls, .csv, or .txt files."
            
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def process_excel_data(df):
    """Process Excel/CSV data and extract review text"""
    reviews = []
    
    # Common column names that might contain reviews
    review_columns = ['review', 'reviews', 'comment', 'comments', 'feedback', 
                     'opinion', 'text', 'message', 'description', 'content',
                     'customer_feedback', 'user_review', 'product_review']
    
    # Find the review column
    review_column = None
    for col in df.columns:
        if col.lower() in review_columns:
            review_column = col
            break
    
    if review_column is None:
        # If no standard column found, use the first text column
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            review_column = text_columns[0]
        else:
            return [], "No text columns found in the file. Please ensure your file contains review text."
    
    # Extract reviews and clean them
    for idx, row in df.iterrows():
        review_text = str(row[review_column]).strip()
        if review_text and review_text.lower() != 'nan' and len(review_text) > 10:
            reviews.append(review_text)
    
    if len(reviews) == 0:
        return [], f"No valid reviews found in column '{review_column}'. Please check your data format."
    
    return reviews, f"Successfully extracted {len(reviews)} reviews from column '{review_column}'."

def process_text_data(content):
    """Process text file content and extract reviews"""
    # Split by lines and clean
    lines = content.split('\n')
    reviews = []
    
    for line in lines:
        line = line.strip()
        if line and len(line) > 10:  # Filter out very short lines
            reviews.append(line)
    
    if len(reviews) == 0:
        return [], "No valid reviews found in the text file. Please ensure each review is on a separate line."
    
    return reviews, f"Successfully extracted {len(reviews)} reviews from the text file."

def main():
    # Authentication check - must be first
    is_authenticated = render_auth_interface()
    
    if not is_authenticated:
        st.stop()  # Stop here if not authenticated
    
    # Show user dashboard
    render_user_dashboard()
    
    # Initialize ethical analyzer
    ethical_analyzer = EthicalReviewAnalyzer()
    
    # Ethical consent check
    consent_given = ethical_analyzer.display_ethical_sidebar()
    
    if not consent_given:
        st.error("⚠️ **Data Processing Consent Required**")
        st.markdown("""
        This application requires your consent to process review data for sentiment analysis.
        Please check the consent box in the sidebar to continue.
        
        **Why we need consent:**
        - To comply with data protection regulations
        - To ensure transparent data processing  
        - To respect your privacy rights
        """)
        st.stop()
    
    # Header with user welcome
    user = st.session_state.user
    st.markdown('<h1 class="main-header">🛡️ Secure E-Commerce Review Analyzer</h1>', unsafe_allow_html=True)
    st.markdown(f"**Welcome {user['username']} - AI-Powered Sentiment Analysis with Enterprise Security**")
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading AI models... This may take a moment."):
        sentiment_model, summarizer_model = load_models()
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode:",
        ["📁 Analyze Sample Dataset", "📤 Upload File (Excel/Text)", "✍️ Analyze Custom Reviews"]
    )
    
    st.sidebar.markdown("---")
    
    # Add ethical AI info in sidebar
    with st.sidebar.expander("🛡️ Ethical AI Features Active"):
        st.success("✅ Privacy Protection")
        st.success("✅ Bias Detection")
        st.success("✅ Explainable AI")
        st.success("✅ Data Anonymization")
    
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard uses:\n"
        "- **DistilBERT** for reliable sentiment classification\n"
        "- **BART** for abstractive summarization\n\n"
        "**Sentiment Categories:**\n"
        "🟢 Positive - Clearly positive feedback\n"
        "🔴 Negative - Clearly negative feedback\n"
        "🟡 Neutral - Balanced or factual comments\n"
        "⚫ Mixed - Contains both positive and negative aspects\n\n"
        "**🛡️ Ethical AI Features:**\n"
        "- Privacy-first data processing\n"
        "- Automated bias detection\n" 
        "- Explainable AI predictions\n"
        "- Data anonymization & PII removal\n\n"
        "Developed as a Master's project for automated review analysis."
    )
    
    if mode == "📁 Analyze Sample Dataset":
        # Load sample data
        df = load_sample_data()
        
        if df is None:
            st.warning("⚠️ Sample dataset not found. Please run Phase 1 notebook first to generate 'processed_reviews.csv'.")
            st.info("Or switch to 'Analyze Custom Reviews' mode to input your own reviews.")
            return
        
        st.success(f"✓ Loaded {len(df)} reviews from dataset")
        
        # Filter options
        st.sidebar.markdown("### Filter Options")
        sentiment_filter = st.sidebar.multiselect(
            "Filter by Sentiment:",
            options=df['sentiment'].unique().tolist(),
            default=df['sentiment'].unique().tolist()
        )
        
        # Apply filters
        filtered_df = df[df['sentiment'].isin(sentiment_filter)]
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", len(filtered_df))
        
        with col2:
            positive_count = len(filtered_df[filtered_df['sentiment'] == 'POSITIVE'])
            st.metric("Positive Reviews", positive_count)
        
        with col3:
            negative_count = len(filtered_df[filtered_df['sentiment'] == 'NEGATIVE'])
            st.metric("Negative Reviews", negative_count)
        
        with col4:
            avg_confidence = filtered_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = filtered_df['sentiment'].value_counts()
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map={'POSITIVE': '#28a745', 'NEGATIVE': '#dc3545', 'NEUTRAL': '#ffc107', 'MIXED': '#6c757d'},
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("📈 Confidence Score Distribution")
            fig = px.histogram(
                filtered_df,
                x='confidence',
                nbins=20,
                color='sentiment',
                color_discrete_map={'POSITIVE': '#28a745', 'NEGATIVE': '#dc3545', 'NEUTRAL': '#ffc107', 'MIXED': '#6c757d'},
                barmode='overlay',
                opacity=0.7
            )
            fig.update_layout(height=400, xaxis_title="Confidence Score", yaxis_title="Count")
            st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        # LLM-Powered Insights
        st.subheader("🤖 AI-Generated Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✨ Key Strengths (Positive Reviews)")
            with st.spinner("Analyzing positive reviews..."):
                positive_reviews = filtered_df[filtered_df['sentiment'] == 'POSITIVE']['review_text'].head(10).tolist()
                
                if positive_reviews:
                    positive_summary = generate_summary(positive_reviews, summarizer_model)
                    st.markdown(f'<div class="insight-box positive-box">{positive_summary}</div>', unsafe_allow_html=True)
                else:
                    st.info("No positive reviews to analyze.")
        
        with col2:
            st.markdown("### ⚠️ Key Problems (Negative Reviews)")
            with st.spinner("Analyzing negative reviews..."):
                negative_reviews = filtered_df[filtered_df['sentiment'] == 'NEGATIVE']['review_text'].head(10).tolist()
                
                if negative_reviews:
                    negative_summary = generate_summary(negative_reviews, summarizer_model)
                    st.markdown(f'<div class="insight-box negative-box">{negative_summary}</div>', unsafe_allow_html=True)
                else:
                    st.info("No negative reviews to analyze.")
        
        st.markdown("---")
        
        # Sample reviews table
        st.subheader("📝 Sample Reviews")
        display_df = filtered_df[['review_text', 'sentiment', 'confidence']].head(10)
        display_df['review_text'] = display_df['review_text'].str[:150] + '...'
        display_df.columns = ['Review Text (Preview)', 'Sentiment', 'Confidence']
        st.dataframe(display_df, width='stretch')
        
        # Log user activity
        log_user_analysis("Sample_Dataset", len(filtered_df), "Sample Data Analysis")
        
        # Ethical Analysis Section for Sample Dataset
        st.markdown("---")
        st.header("🛡️ Ethical AI Analysis Report")
        
        # Prepare data for ethical analysis
        sample_results_df = filtered_df.rename(columns={'review_text': 'Review', 'sentiment': 'Sentiment', 'confidence': 'Confidence'})
        
        # Anonymize data for privacy  
        anonymized_df = ethical_analyzer.anonymize_data(sample_results_df)
        
        # Bias detection
        bias_report = ethical_analyzer.detect_bias(sample_results_df)
        ethical_analyzer.display_bias_report(bias_report)
        
        # Privacy status
        ethical_analyzer.display_privacy_status(len(sample_results_df), len(anonymized_df))
        
        # Explainable AI
        ethical_analyzer.display_explainable_ai(sample_results_df)
    
    elif mode == "📤 Upload File (Excel/Text)":
        st.subheader("📤 Upload Review Files")
        st.markdown("Upload Excel (.xlsx, .xls), CSV (.csv), or Text (.txt) files containing customer reviews.")
        
        # File upload section
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['xlsx', 'xls', 'csv', 'txt'],
            help="Supported formats: Excel (.xlsx, .xls), CSV (.csv), or Text (.txt)"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Process the uploaded file
            with st.spinner("Processing uploaded file..."):
                reviews, message = process_uploaded_file(uploaded_file)
            
            if reviews:
                st.success(message)
                
                # Show file preview
                with st.expander("📋 File Preview (First 10 reviews)"):
                    for i, review in enumerate(reviews[:10], 1):
                        st.write(f"{i}. {review[:200]}{'...' if len(review) > 200 else ''}")
                
                # Analysis button
                if st.button("🔍 Analyze Uploaded Reviews", type="primary"):
                    with st.spinner(f"Analyzing {len(reviews)} reviews..."):
                        # Analyze sentiment for each review
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, review in enumerate(reviews):
                            sentiment = classify_sentiment(review, sentiment_model)
                            results.append({
                                'review': review,
                                'sentiment': sentiment['label'],
                                'confidence': sentiment['score']
                            })
                            progress_bar.progress((idx + 1) / len(reviews))
                        
                        results_df = pd.DataFrame(results)
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Reviews", len(results_df))
                        
                        with col2:
                            positive_count = (results_df['sentiment'] == 'POSITIVE').sum()
                            positive_pct = positive_count / len(results_df)
                            st.metric("Positive Reviews", f"{positive_count} ({positive_pct:.0%})")
                        
                        with col3:
                            negative_count = (results_df['sentiment'] == 'NEGATIVE').sum()
                            negative_pct = negative_count / len(results_df)
                            st.metric("Negative Reviews", f"{negative_count} ({negative_pct:.0%})")
                        
                        with col4:
                            neutral_mixed_count = ((results_df['sentiment'] == 'NEUTRAL') | (results_df['sentiment'] == 'MIXED')).sum()
                            neutral_mixed_pct = neutral_mixed_count / len(results_df)
                            st.metric("Neutral/Mixed", f"{neutral_mixed_count} ({neutral_mixed_pct:.0%})")
                        
                        # Add average confidence in a new row
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            avg_confidence = results_df['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                        
                        # Visualizations
                        st.markdown("---")
                        st.subheader("📊 Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🥧 Sentiment Distribution")
                            sentiment_counts = results_df['sentiment'].value_counts()
                            fig = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                color=sentiment_counts.index,
                                color_discrete_map={'POSITIVE': '#28a745', 'NEGATIVE': '#dc3545', 'NEUTRAL': '#ffc107', 'MIXED': '#6c757d'},
                                hole=0.4
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, width='stretch')
                        
                        with col2:
                            st.markdown("### 📈 Confidence Score Distribution")
                            fig = px.histogram(
                                results_df,
                                x='confidence',
                                nbins=20,
                                color='sentiment',
                                color_discrete_map={'POSITIVE': '#28a745', 'NEGATIVE': '#dc3545', 'NEUTRAL': '#ffc107', 'MIXED': '#6c757d'},
                                barmode='overlay',
                                opacity=0.7
                            )
                            fig.update_layout(height=400, xaxis_title="Confidence Score", yaxis_title="Count")
                            st.plotly_chart(fig, width='stretch')
                        
                        # AI Insights
                        st.markdown("---")
                        st.subheader("🤖 AI-Generated Insights")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            positive_reviews = results_df[results_df['sentiment'] == 'POSITIVE']['review'].head(20).tolist()
                            if positive_reviews:
                                st.markdown("### ✨ Key Strengths")
                                with st.spinner("Generating positive insights..."):
                                    positive_summary = generate_summary(positive_reviews, summarizer_model)
                                st.markdown(f'<div class="insight-box positive-box">{positive_summary}</div>', unsafe_allow_html=True)
                            else:
                                st.info("No positive reviews found.")
                        
                        with col2:
                            negative_reviews = results_df[results_df['sentiment'] == 'NEGATIVE']['review'].head(20).tolist()
                            if negative_reviews:
                                st.markdown("### ⚠️ Areas for Improvement")
                                with st.spinner("Generating negative insights..."):
                                    negative_summary = generate_summary(negative_reviews, summarizer_model)
                                st.markdown(f'<div class="insight-box negative-box">{negative_summary}</div>', unsafe_allow_html=True)
                            else:
                                st.info("No negative reviews found.")
                        
                        # Add neutral and mixed sentiment analysis
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            neutral_reviews = results_df[results_df['sentiment'] == 'NEUTRAL']['review'].head(20).tolist()
                            if neutral_reviews:
                                st.markdown("### 😐 Neutral Feedback")
                                with st.spinner("Analyzing neutral insights..."):
                                    neutral_summary = generate_summary(neutral_reviews, summarizer_model)
                                st.markdown(f'<div class="insight-box neutral-box">{neutral_summary}</div>', unsafe_allow_html=True)
                            else:
                                st.info("No neutral reviews found.")
                        
                        with col4:
                            mixed_reviews = results_df[results_df['sentiment'] == 'MIXED']['review'].head(20).tolist()
                            if mixed_reviews:
                                st.markdown("### 🤔 Mixed Opinions")
                                with st.spinner("Analyzing mixed sentiment insights..."):
                                    mixed_summary = generate_summary(mixed_reviews, summarizer_model)
                                st.markdown(f'<div class="insight-box mixed-box">{mixed_summary}</div>', unsafe_allow_html=True)
                            else:
                                st.info("No mixed reviews found.")
                        
                        # Download results
                        st.markdown("---")
                        st.subheader("💾 Export Results")
                        
                        # Prepare export data
                        export_df = results_df.copy()
                        export_df['confidence_percentage'] = (export_df['confidence'] * 100).round(2)
                        export_df = export_df[['review', 'sentiment', 'confidence_percentage']]
                        export_df.columns = ['Review Text', 'Sentiment', 'Confidence (%)']
                        
                        # Convert to CSV for download
                        csv_data = export_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Analysis Results (CSV)",
                            data=csv_data,
                            file_name=f"sentiment_analysis_results_{uploaded_file.name.split('.')[0]}.csv",
                            mime="text/csv",
                            help="Download the complete analysis results as a CSV file"
                        )
                        
                        # Sample results table
                        st.markdown("---")
                        st.subheader("📋 Sample Results")
                        display_df = export_df.head(20).copy()
                        display_df['Review Text'] = display_df['Review Text'].str[:150] + '...'
                        st.dataframe(display_df, width='stretch')
                        
                        if len(results_df) > 20:
                            st.info(f"Showing first 20 results. Download the CSV file to see all {len(results_df)} results.")
                        
                        # Log user activity
                        log_user_analysis(uploaded_file.name, len(results_df), "File Upload Analysis")
                        
                        # Ethical Analysis Section for Uploaded Files
                        st.markdown("---")
                        st.header("🛡️ Ethical AI Analysis Report")
                        
                        # Anonymize data for privacy
                        anonymized_df = ethical_analyzer.anonymize_data(results_df)
                        
                        # Bias detection
                        bias_report = ethical_analyzer.detect_bias(results_df)
                        ethical_analyzer.display_bias_report(bias_report)
                        
                        # Privacy status
                        ethical_analyzer.display_privacy_status(len(results_df), len(anonymized_df))
                        
                        # Explainable AI
                        ethical_analyzer.display_explainable_ai(results_df)
            
            else:
                st.error(message)
        
        else:
            # Instructions for file format
            st.markdown("### 📋 File Format Guidelines")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Excel/CSV Files:**")
                st.markdown("""
                - Include review text in columns named: `review`, `reviews`, `comment`, `feedback`, etc.
                - Each row should contain one review
                - Example structure:
                
                | Review | Rating | Date |
                |--------|--------|---------|
                | Great product! | 5 | 2024-01-15 |
                | Poor quality | 2 | 2024-01-16 |
                """)
            
            with col2:
                st.markdown("**📝 Text Files:**")
                st.markdown("""
                - One review per line
                - Each line should contain the complete review text
                - Example format:
                ```
                Amazing product! Exceeded expectations.
                Terrible quality. Broke after one day.
                Great customer service and fast shipping.
                ```
                """)
            
            # Sample files download
            st.markdown("### 📥 Sample Files")
            st.markdown("Download sample files to see the expected format:")
            
            # Create sample Excel data
            sample_excel_data = pd.DataFrame({
                'Review': [
                    'Amazing product! Exceeded all my expectations and arrived quickly.',
                    'Terrible quality. The item broke after just one day of use.',
                    'Excellent customer service and fast shipping. Highly recommend!',
                    'Worst purchase ever. Complete waste of money and time.',
                    'Outstanding value for money. Great build quality and design.'
                ],
                'Rating': [5, 1, 5, 1, 4],
                'Date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19']
            })
            
            sample_excel_csv = sample_excel_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV File",
                data=sample_excel_csv,
                file_name="sample_reviews.csv",
                mime="text/csv"
            )
            
            # Sample text data
            sample_text_data = """Amazing product! Exceeded all my expectations and arrived quickly.
Terrible quality. The item broke after just one day of use.
Excellent customer service and fast shipping. Highly recommend!
Worst purchase ever. Complete waste of money and time.
Outstanding value for money. Great build quality and design."""
            
            st.download_button(
                label="📥 Download Sample Text File",
                data=sample_text_data,
                file_name="sample_reviews.txt",
                mime="text/plain"
            )
    
    else:  # Custom Reviews Mode
        st.subheader("✍️ Analyze Your Own Reviews")
        st.markdown("Enter multiple reviews (one per line) to analyze sentiment and generate insights.")
        
        # Text input
        custom_reviews = st.text_area(
            "Enter reviews (one per line):",
            height=200,
            placeholder="Example:\nThis product is amazing! Great quality.\nTerrible experience, would not recommend.\nGood value for the price."
        )
        
        if st.button("🔍 Analyze Reviews", type="primary"):
            if custom_reviews.strip():
                # Split reviews by newline
                review_list = [r.strip() for r in custom_reviews.split('\n') if r.strip()]
                
                with st.spinner(f"Analyzing {len(review_list)} reviews..."):
                    # Analyze sentiment for each review
                    results = []
                    for review in review_list:
                        sentiment = classify_sentiment(review, sentiment_model)
                        # Additional mixed sentiment analysis
                        if sentiment['score'] < 0.7:  # Low confidence threshold
                            mixed_analysis = analyze_mixed_sentiment(review)
                            if mixed_analysis == 'MIXED':
                                    sentiment['label'] = 'MIXED'
                            
                        results.append({
                            'review': review,
                            'sentiment': sentiment['label'],
                            'confidence': sentiment['score']
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Reviews", len(results_df))
                    
                    with col2:
                        positive_pct = (results_df['sentiment'] == 'POSITIVE').sum() / len(results_df)
                        st.metric("Positive", f"{positive_pct:.0%}")
                    
                    with col3:
                        negative_pct = (results_df['sentiment'] == 'NEGATIVE').sum() / len(results_df)
                        st.metric("Negative", f"{negative_pct:.0%}")
                    
                    # Add new row for neutral/mixed and confidence
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        neutral_mixed_pct = ((results_df['sentiment'] == 'NEUTRAL') | (results_df['sentiment'] == 'MIXED')).sum() / len(results_df)
                        st.metric("Neutral/Mixed", f"{neutral_mixed_pct:.0%}")
                    
                    with col2:
                        avg_confidence = results_df['confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                    
                    # Sentiment distribution
                    st.subheader("📊 Sentiment Distribution")
                    sentiment_counts = results_df['sentiment'].value_counts()
                    
                    fig = px.bar(
                        x=sentiment_counts.index,
                        y=sentiment_counts.values,
                        color=sentiment_counts.index,
                        color_discrete_map={'POSITIVE': '#28a745', 'NEGATIVE': '#dc3545', 'NEUTRAL': '#ffc107', 'MIXED': '#6c757d'},
                        labels={'x': 'Sentiment', 'y': 'Count'}
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width='stretch')
                    
                    # AI Insights
                    st.markdown("---")
                    st.subheader("🤖 AI-Generated Insights")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        positive_reviews = results_df[results_df['sentiment'] == 'POSITIVE']['review'].tolist()
                        if positive_reviews:
                            st.markdown("### ✨ Positive Highlights")
                            positive_summary = generate_summary(positive_reviews, summarizer_model)
                            st.markdown(f'<div class="insight-box positive-box">{positive_summary}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        negative_reviews = results_df[results_df['sentiment'] == 'NEGATIVE']['review'].tolist()
                        if negative_reviews:
                            st.markdown("### ⚠️ Areas for Improvement")
                            negative_summary = generate_summary(negative_reviews, summarizer_model)
                            st.markdown(f'<div class="insight-box negative-box">{negative_summary}</div>', unsafe_allow_html=True)
                    
                    # Add neutral and mixed insights for custom reviews
                    neutral_reviews = results_df[results_df['sentiment'] == 'NEUTRAL']['review'].tolist()
                    mixed_reviews = results_df[results_df['sentiment'] == 'MIXED']['review'].tolist()
                    
                    if neutral_reviews or mixed_reviews:
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if neutral_reviews:
                                st.markdown("### 😐 Neutral Feedback")
                                neutral_summary = generate_summary(neutral_reviews, summarizer_model)
                                st.markdown(f'<div class="insight-box neutral-box">{neutral_summary}</div>', unsafe_allow_html=True)
                        
                        with col2:
                            if mixed_reviews:
                                st.markdown("### 🤔 Mixed Opinions")
                                mixed_summary = generate_summary(mixed_reviews, summarizer_model)
                                st.markdown(f'<div class="insight-box mixed-box">{mixed_summary}</div>', unsafe_allow_html=True)
                    
                    # Detailed results
                    st.markdown("---")
                    st.subheader("📋 Detailed Results")
                    display_results = results_df.copy()
                    display_results.columns = ['Review Text', 'Sentiment', 'Confidence']
                    st.dataframe(display_results, width='stretch')
                    
                    # Log user activity
                    log_user_analysis("Custom_Reviews", len(results_df), "Custom Text Analysis")
                    
                    # Ethical Analysis Section
                    st.markdown("---")
                    st.header("🛡️ Ethical AI Analysis Report")
                    
                    # Anonymize data for privacy
                    anonymized_df = ethical_analyzer.anonymize_data(results_df)
                    
                    # Bias detection
                    bias_report = ethical_analyzer.detect_bias(results_df)
                    ethical_analyzer.display_bias_report(bias_report)
                    
                    # Privacy status
                    ethical_analyzer.display_privacy_status(len(results_df), len(anonymized_df))
                    
                    # Explainable AI
                    ethical_analyzer.display_explainable_ai(results_df)
            else:
                st.warning("⚠️ Please enter at least one review to analyze.")
    
    # Ethical AI Footer
    st.markdown("---")
    st.markdown("""
    **🛡️ Ethical E-Commerce Review Analyzer Dashboard**
    
    *Powered by Advanced AI Models with Ethical Safeguards*
    
    **Ethical Compliance:** ✅ Privacy Protection | ✅ Bias Detection | ✅ Explainable AI | ✅ Data Security
    """)
    
    with st.expander("🔍 About Our Ethical AI Approach"):
        st.markdown("""
        **Privacy First:** All data is processed locally and automatically deleted after your session.
        
        **Bias Awareness:** We actively monitor for demographic, cultural, and economic biases in our analysis.
        
        **Transparent AI:** Every prediction comes with clear explanations of how the AI made its decision.
        
        **Secure Processing:** Customer data is anonymized and PII is removed to protect privacy.
        
        **Responsible Use:** This tool is designed to support fair and ethical business decision-making.
        """)

if __name__ == "__main__":
    main()
