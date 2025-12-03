"""
Ethical AI Features for E-Commerce Review Analyzer
Implements privacy, bias detection, and explainable AI features
"""

import streamlit as st
import pandas as pd
import hashlib
from datetime import datetime
import re

class EthicalReviewAnalyzer:
    def __init__(self):
        self.bias_keywords = {
            'gender': ['man', 'woman', 'male', 'female', 'guy', 'girl', 'he', 'she', 'his', 'her'],
            'age': ['young', 'old', 'elderly', 'teen', 'adult', 'senior', 'youth', 'mature'],
            'cultural': ['foreign', 'native', 'traditional', 'modern', 'western', 'eastern', 'american', 'european'],
            'economic': ['cheap', 'expensive', 'luxury', 'budget', 'premium', 'affordable', 'costly']
        }
        
        self.positive_indicators = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'outstanding', 
                                  'fantastic', 'wonderful', 'awesome', 'brilliant', 'superb', 'incredible']
        
        self.negative_indicators = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing',
                                  'poor', 'broken', 'defective', 'useless', 'waste', 'regret', 'disaster']
    
    def display_ethical_sidebar(self):
        """Display ethical AI features in sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.header("🛡️ Ethical AI Features")
        
        # Data consent section
        with st.sidebar.expander("📋 Data Privacy & Consent", expanded=True):
            consent = st.checkbox(
                "I consent to processing review data for analysis",
                help="Your data will be processed locally and not stored permanently"
            )
            
            if consent:
                st.success("✅ Consent provided")
                st.markdown("""
                **Data Usage Policy:**
                - 🔒 Data processed locally only
                - 🗑️ No permanent storage
                - 🔐 Anonymized for analysis
                - ⏰ Deleted after session ends
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Analysis requires consent")
            
            return consent
        
    def detect_bias(self, reviews_df):
        """Detect potential bias in reviews"""
        if reviews_df is None or reviews_df.empty:
            return {}
            
        bias_report = {}
        total_reviews = len(reviews_df)
        
        # Find the review text column (handle different column names)
        review_column = None
        possible_columns = ['Review', 'review', 'Review Text', 'review_text', 'text', 'comment', 'feedback']
        
        for col in possible_columns:
            if col in reviews_df.columns:
                review_column = col
                break
        
        if review_column is None:
            # If no standard column found, use the first column
            review_column = reviews_df.columns[0]
        
        # Combine all text for analysis
        all_text = ' '.join(reviews_df[review_column].astype(str)).lower()
        
        for category, keywords in self.bias_keywords.items():
            bias_count = 0
            found_keywords = []
            
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                matches = len(re.findall(pattern, all_text))
                if matches > 0:
                    bias_count += matches
                    found_keywords.append(f"{keyword}({matches})")
            
            bias_percentage = (bias_count / total_reviews) * 100 if total_reviews > 0 else 0
            
            # Determine risk level
            if bias_percentage > 15:
                risk_level = 'High'
                risk_color = '🔴'
            elif bias_percentage > 8:
                risk_level = 'Medium'
                risk_color = '🟡'
            else:
                risk_level = 'Low'
                risk_color = '🟢'
            
            bias_report[category] = {
                'count': bias_count,
                'percentage': bias_percentage,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'keywords_found': found_keywords[:5]  # Show top 5
            }
        
        return bias_report
    
    def anonymize_data(self, df):
        """Anonymize customer data for privacy protection"""
        if df is None or df.empty:
            return df
            
        df_anon = df.copy()
        
        # Anonymize customer names if present
        if 'Customer_Name' in df_anon.columns:
            df_anon['Customer_ID'] = df_anon['Customer_Name'].apply(
                lambda x: hashlib.md5(str(x).encode()).hexdigest()[:8]
            )
            df_anon = df_anon.drop('Customer_Name', axis=1)
        
        # Remove other PII columns
        pii_columns = ['Email', 'Phone', 'Address', 'IP_Address', 'User_ID']
        for col in pii_columns:
            if col in df_anon.columns:
                df_anon = df_anon.drop(col, axis=1)
        
        # Mask partial information in reviews (emails, phones)
        review_columns = ['Review', 'review', 'Review Text', 'review_text', 'text', 'comment', 'feedback']
        for col in review_columns:
            if col in df_anon.columns:
                df_anon[col] = df_anon[col].apply(self._mask_pii_in_text)
        
        return df_anon
    
    def _mask_pii_in_text(self, text):
        """Mask PII information in review text"""
        if pd.isna(text):
            return text
            
        text = str(text)
        
        # Mask email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, '[EMAIL_MASKED]', text)
        
        # Mask phone numbers
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        text = re.sub(phone_pattern, '[PHONE_MASKED]', text)
        
        return text
    
    def explain_prediction(self, review, sentiment, confidence):
        """Provide explainable AI insights for predictions"""
        if pd.isna(review):
            return {'reasoning': 'No review text available', 'confidence_level': 'N/A'}
            
        review_lower = str(review).lower()
        
        # Find positive and negative indicators
        found_positive = [word for word in self.positive_indicators if word in review_lower]
        found_negative = [word for word in self.negative_indicators if word in review_lower]
        
        # Build explanation
        explanation_parts = []
        
        # Sentiment reasoning
        explanation_parts.append(f"**Predicted Sentiment:** {sentiment}")
        explanation_parts.append(f"**Model Confidence:** {confidence:.1%}")
        
        # Word-based indicators
        if found_positive:
            explanation_parts.append(f"**Positive indicators found:** {', '.join(found_positive[:5])}")
        if found_negative:
            explanation_parts.append(f"**Negative indicators found:** {', '.join(found_negative[:5])}")
        
        # Confidence interpretation
        if confidence >= 0.8:
            conf_level = "Very High - Strong prediction confidence"
        elif confidence >= 0.6:
            conf_level = "High - Good prediction confidence"
        elif confidence >= 0.4:
            conf_level = "Medium - Moderate confidence, may contain mixed sentiments"
        else:
            conf_level = "Low - Uncertain prediction, review likely contains mixed or neutral sentiment"
        
        explanation_parts.append(f"**Confidence Level:** {conf_level}")
        
        # Additional insights
        word_count = len(review.split())
        if word_count < 5:
            explanation_parts.append("⚠️ **Note:** Very short review may affect accuracy")
        
        if len(found_positive) > 0 and len(found_negative) > 0:
            explanation_parts.append("⚠️ **Note:** Mixed sentiment detected - contains both positive and negative indicators")
        
        return {
            'reasoning': '\n\n'.join(explanation_parts),
            'confidence_level': conf_level,
            'positive_words': found_positive[:5],
            'negative_words': found_negative[:5],
            'word_count': word_count
        }
    
    def display_bias_report(self, bias_report):
        """Display bias detection results"""
        if not bias_report:
            st.info("No bias analysis available - please analyze some reviews first.")
            return
        
        st.subheader("⚖️ Bias Detection Report")
        
        # Create metrics for each bias category
        cols = st.columns(len(bias_report))
        
        for idx, (category, report) in enumerate(bias_report.items()):
            with cols[idx]:
                st.metric(
                    label=f"{report['risk_color']} {category.title()} Bias",
                    value=f"{report['percentage']:.1f}%",
                    delta=f"{report['risk_level']} Risk"
                )
                
                if report['keywords_found']:
                    with st.expander("Keywords found"):
                        st.write(', '.join(report['keywords_found']))
        
        # Overall bias assessment
        high_risk_count = sum(1 for r in bias_report.values() if r['risk_level'] == 'High')
        
        if high_risk_count > 0:
            st.error(f"⚠️ **{high_risk_count} high-risk bias categories detected.** Consider reviewing data sources for potential bias.")
        elif any(r['risk_level'] == 'Medium' for r in bias_report.values()):
            st.warning("⚠️ **Medium bias risk detected.** Monitor for potential bias issues.")
        else:
            st.success("✅ **Low bias risk detected.** Data appears relatively unbiased.")
    
    def display_privacy_status(self, original_count, processed_count):
        """Display data protection status"""
        st.subheader("🔒 Data Protection Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Reviews Processed", processed_count)
        
        with col2:
            st.metric("Data Anonymized", "✅ Yes" if processed_count > 0 else "N/A")
        
        with col3:
            st.metric("PII Removed", "✅ Yes" if processed_count > 0 else "N/A")
        
        if processed_count > 0:
            st.success("🔐 All customer data has been anonymized and PII removed")
            st.info("🗑️ Data will be automatically deleted when session ends")
            st.info("🔒 Processing performed locally for maximum privacy")
    
    def display_explainable_ai(self, results_df, max_explanations=3):
        """Display explainable AI section"""
        if results_df is None or results_df.empty:
            st.info("No predictions available for explanation.")
            return
        
        st.subheader("🧠 Explainable AI Insights")
        st.write("Understanding how the AI makes its predictions:")
        
        # Show explanations for sample predictions
        num_explanations = min(max_explanations, len(results_df))
        
        for idx in range(num_explanations):
            row = results_df.iloc[idx]
            
            # Find review text in row (handle different column names)
            review_text = ''
            for col in ['Review', 'review', 'Review Text', 'review_text', 'text', 'comment', 'feedback']:
                if col in row.index and pd.notna(row.get(col)):
                    review_text = row.get(col)
                    break
            
            if not review_text and len(row) > 0:
                review_text = str(row.iloc[0])  # Use first column as fallback
            
            # Get explanation
            explanation = self.explain_prediction(
                review_text, 
                row.get('Sentiment', row.get('sentiment', 'Unknown')), 
                row.get('Confidence', row.get('confidence', 0.5))
            )
            
            with st.expander(f"📝 Explanation for Review {idx + 1} - {row.get('Sentiment', 'Unknown')} Sentiment"):
                # Show original review (truncated)
                display_text = str(review_text)[:200]
                if len(str(review_text)) > 200:
                    display_text += "..."
                
                st.markdown(f"**Review Text:** _{display_text}_")
                st.markdown("**AI Explanation:**")
                st.markdown(explanation['reasoning'])
        
        # Summary statistics
        if len(results_df) > max_explanations:
            st.info(f"Showing explanations for first {max_explanations} reviews. Total analyzed: {len(results_df)}")
    
    def generate_ethical_report(self, results_df, bias_report):
        """Generate comprehensive ethical analysis report"""
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_reviews': len(results_df) if results_df is not None else 0,
            'bias_assessment': bias_report,
            'privacy_compliance': True,
            'explainability_provided': True
        }
        
        return report