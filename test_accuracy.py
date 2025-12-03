"""
ML Accuracy Assessment Script
Phase 2: Testing the sentiment analysis model

This script evaluates the sentiment classification model's accuracy
on a manually labeled test dataset of 100 reviews.
"""

import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_test_data(filepath='test_dataset.csv'):
    """Load the manually labeled test dataset"""
    df = pd.read_csv(filepath)
    return df

def load_sentiment_model():
    """Load the sentiment analysis model"""
    print("Loading sentiment analysis model...")
    model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )
    print("✓ Model loaded successfully!")
    return model

def classify_sentiment(text, model):
    """Classify sentiment using the model"""
    try:
        result = model(text)[0]
        return result['label'], result['score']
    except Exception as e:
        return 'NEUTRAL', 0.0

def evaluate_model(test_df, model):
    """Evaluate model performance on test dataset"""
    print("\nEvaluating model on test dataset...")
    
    # Predict sentiments
    predictions = []
    confidences = []
    
    for idx, row in test_df.iterrows():
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} reviews...")
        
        label, score = classify_sentiment(row['review_text'], model)
        predictions.append(label)
        confidences.append(score)
    
    test_df['predicted_label'] = predictions
    test_df['confidence'] = confidences
    
    # Convert labels to match format
    # Model outputs: POSITIVE, NEGATIVE
    # Manual labels: positive, negative
    test_df['predicted_label_normalized'] = test_df['predicted_label'].str.lower()
    test_df['manual_label_normalized'] = test_df['manual_label'].str.lower()
    
    return test_df

def calculate_metrics(test_df):
    """Calculate and display accuracy metrics"""
    y_true = test_df['manual_label_normalized']
    y_pred = test_df['predicted_label_normalized']
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*80)
    print("ML ACCURACY ASSESSMENT RESULTS")
    print("="*80)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nDetailed Classification Report:")
    print("-"*80)
    print(classification_report(y_true, y_pred, target_names=['negative', 'positive']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['negative', 'positive'])
    
    return accuracy, cm

def visualize_results(test_df, cm):
    """Create visualizations of the results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # 2. Prediction Distribution
    pred_counts = test_df['predicted_label'].value_counts()
    axes[0, 1].bar(pred_counts.index, pred_counts.values, color=['#dc3545', '#28a745'], alpha=0.7)
    axes[0, 1].set_title('Predicted Sentiment Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Sentiment')
    axes[0, 1].set_ylabel('Count')
    
    # 3. Confidence Score Distribution
    axes[1, 0].hist(test_df['confidence'], bins=20, color='#4c6ef5', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(test_df['confidence'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {test_df["confidence"].mean():.3f}')
    axes[1, 0].set_title('Model Confidence Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Confidence Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Confidence by Correctness
    test_df['correct'] = test_df['manual_label_normalized'] == test_df['predicted_label_normalized']
    correct_df = test_df[test_df['correct']]
    incorrect_df = test_df[~test_df['correct']]
    
    axes[1, 1].boxplot(
        [correct_df['confidence'], incorrect_df['confidence']],
        labels=['Correct', 'Incorrect'],
        patch_artist=True,
        boxprops=dict(facecolor='#51cf66', alpha=0.7)
    )
    axes[1, 1].set_title('Confidence: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Confidence Score')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('accuracy_assessment_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'accuracy_assessment_results.png'")
    plt.show()

def analyze_errors(test_df):
    """Analyze misclassified examples"""
    incorrect_predictions = test_df[
        test_df['manual_label_normalized'] != test_df['predicted_label_normalized']
    ]
    
    if len(incorrect_predictions) > 0:
        print("\n" + "="*80)
        print(f"MISCLASSIFIED EXAMPLES ({len(incorrect_predictions)} total)")
        print("="*80)
        
        for idx, row in incorrect_predictions.head(5).iterrows():
            print(f"\nExample {idx + 1}:")
            print(f"Review: {row['review_text']}")
            print(f"True Label: {row['manual_label']}")
            print(f"Predicted: {row['predicted_label']} (Confidence: {row['confidence']:.4f})")
            print("-"*80)
    else:
        print("\n🎉 Perfect accuracy! No misclassifications!")

def save_results(test_df, accuracy):
    """Save detailed results to CSV"""
    test_df.to_csv('accuracy_test_results.csv', index=False)
    print("\n✓ Detailed results saved to 'accuracy_test_results.csv'")
    
    # Save summary
    summary = {
        'Total_Samples': len(test_df),
        'Correct_Predictions': (test_df['manual_label_normalized'] == test_df['predicted_label_normalized']).sum(),
        'Incorrect_Predictions': (test_df['manual_label_normalized'] != test_df['predicted_label_normalized']).sum(),
        'Accuracy': accuracy,
        'Average_Confidence': test_df['confidence'].mean()
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('accuracy_summary.csv', index=False)
    print("✓ Summary saved to 'accuracy_summary.csv'")

def main():
    """Main execution function"""
    print("="*80)
    print("E-COMMERCE REVIEW ANALYZER - ML ACCURACY ASSESSMENT")
    print("="*80)
    
    # Load test data
    print("\nStep 1: Loading test dataset...")
    test_df = load_test_data()
    print(f"✓ Loaded {len(test_df)} manually labeled reviews")
    
    # Load model
    print("\nStep 2: Loading sentiment analysis model...")
    model = load_sentiment_model()
    
    # Evaluate model
    print("\nStep 3: Running predictions...")
    test_df = evaluate_model(test_df, model)
    
    # Calculate metrics
    print("\nStep 4: Calculating accuracy metrics...")
    accuracy, cm = calculate_metrics(test_df)
    
    # Visualize results
    print("\nStep 5: Creating visualizations...")
    visualize_results(test_df, cm)
    
    # Analyze errors
    print("\nStep 6: Analyzing misclassifications...")
    analyze_errors(test_df)
    
    # Save results
    print("\nStep 7: Saving results...")
    save_results(test_df, accuracy)
    
    print("\n" + "="*80)
    print("ASSESSMENT COMPLETE!")
    print("="*80)
    print(f"\n📊 Final Accuracy: {accuracy*100:.2f}%")
    print(f"📈 Average Confidence: {test_df['confidence'].mean():.2%}")
    print("\n✓ All results and visualizations have been saved.")

if __name__ == "__main__":
    main()
