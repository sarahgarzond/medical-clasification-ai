"""
BioBERT-based Medical Literature Classifier
==========================================

This module implements a state-of-the-art medical literature classifier using BioBERT,
a BERT model pre-trained on biomedical text (PubMed abstracts and PMC full-text articles).

Key Features:
- Uses BioBERT for contextual understanding of medical terminology
- Fine-tuned for multi-label classification of medical domains
- Supports both CPU and GPU inference
- Provides attention weights for interpretability
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import pickle
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BioBERTClassifier(nn.Module):
    """
    BioBERT-based classifier for medical literature classification.
    
    Architecture:
    - BioBERT base model (110M parameters)
    - Dropout layer for regularization
    - Linear classification head
    - Sigmoid activation for multi-label classification
    """
    
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1", 
                 num_labels: int = 4, dropout_rate: float = 0.3):
        super(BioBERTClassifier, self).__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        
        # Load BioBERT configuration and model
        self.config = AutoConfig.from_pretrained(model_name)
        self.biobert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize classification head weights"""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask to avoid attention on padding tokens
            token_type_ids: Segment IDs (not used in BioBERT)
            labels: Ground truth labels for training
            
        Returns:
            Dictionary with logits, loss (if labels provided), and hidden states
        """
        # Get BioBERT outputs
        outputs = self.biobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }

class MedicalDataset(torch.utils.data.Dataset):
    """
    Custom dataset for medical literature classification
    """
    
    def __init__(self, texts: List[str], labels: List[List[str]], 
                 tokenizer, max_length: int = 512, label_encoder=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = label_encoder
        
        # Encode labels if encoder is provided
        if self.label_encoder:
            self.encoded_labels = self.label_encoder.transform(labels)
        else:
            self.encoded_labels = labels
            
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.encoded_labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

class BioBERTMedicalClassifier:
    """
    Main classifier class that handles training, evaluation, and inference
    """
    
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1",
                 max_length: int = 512, device: str = None):
        
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize label encoder
        self.label_encoder = MultiLabelBinarizer()
        
        # Model will be initialized during training
        self.model = None
        self.trainer = None
        
        # Metrics storage
        self.training_history = []
        self.evaluation_results = {}
        
        logger.info(f"Initialized BioBERT classifier on {self.device}")
        
    def prepare_data(self, df: pd.DataFrame, text_column: str = 'combined_text', 
                    label_column: str = 'domains') -> Tuple[List[str], List[List[str]]]:
        """
        Prepare data for training/inference
        
        Args:
            df: DataFrame with text and labels
            text_column: Column name containing the text
            label_column: Column name containing the labels
            
        Returns:
            Tuple of (texts, labels)
        """
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # Handle different label formats
        if isinstance(df[label_column].iloc[0], str):
            # If labels are strings, split by delimiter
            labels = [label.split('|') if '|' in label else [label] 
                     for label in df[label_column].fillna('').astype(str)]
        else:
            # If labels are already lists
            labels = df[label_column].tolist()
            
        # Clean labels
        labels = [[label.strip() for label in label_list if label.strip()] 
                 for label_list in labels]
        
        return texts, labels
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None,
              text_column: str = 'combined_text', label_column: str = 'domains',
              output_dir: str = 'models/biobert_medical',
              num_epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        """
        Train the BioBERT classifier
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame (optional)
            text_column: Column name containing the text
            label_column: Column name containing the labels
            output_dir: Directory to save the model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
        """
        logger.info("Starting BioBERT training...")
        
        # Prepare training data
        train_texts, train_labels = self.prepare_data(train_df, text_column, label_column)
        
        # Fit label encoder
        self.label_encoder.fit(train_labels)
        num_labels = len(self.label_encoder.classes_)
        
        logger.info(f"Found {num_labels} unique labels: {self.label_encoder.classes_}")
        
        # Initialize model
        self.model = BioBERTClassifier(
            model_name=self.model_name,
            num_labels=num_labels
        ).to(self.device)
        
        # Create datasets
        train_dataset = MedicalDataset(
            train_texts, train_labels, self.tokenizer, 
            self.max_length, self.label_encoder
        )
        
        val_dataset = None
        if val_df is not None:
            val_texts, val_labels = self.prepare_data(val_df, text_column, label_column)
            val_dataset = MedicalDataset(
                val_texts, val_labels, self.tokenizer,
                self.max_length, self.label_encoder
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            learning_rate=learning_rate,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if val_dataset else None,
        )
        
        # Train model
        logger.info("Training started...")
        train_result = self.trainer.train()
        
        # Save model and tokenizer
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label encoder
        with open(f'{output_dir}/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
        # Save training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'train_loss': train_result.training_loss,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        })
        
        logger.info(f"Training completed! Model saved to {output_dir}")
        return train_result
    
    def evaluate(self, test_df: pd.DataFrame, text_column: str = 'combined_text',
                label_column: str = 'domains') -> Dict:
        """
        Evaluate the trained model
        
        Args:
            test_df: Test DataFrame
            text_column: Column name containing the text
            label_column: Column name containing the labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        logger.info("Evaluating model...")
        
        # Prepare test data
        test_texts, test_labels = self.prepare_data(test_df, text_column, label_column)
        
        # Get predictions
        predictions = self.predict(test_texts)
        
        # Convert to binary format for evaluation
        y_true = self.label_encoder.transform(test_labels)
        y_pred = np.array([pred['binary_predictions'] for pred in predictions])
        
        # Calculate metrics
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix (for each class)
        confusion_matrices = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            confusion_matrices[class_name] = cm.tolist()
        
        self.evaluation_results = {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': report,
            'confusion_matrices': confusion_matrices,
            'num_samples': len(test_texts)
        }
        
        logger.info(f"Evaluation completed. F1-weighted: {f1_weighted:.4f}")
        return self.evaluation_results
    
    def predict(self, texts: List[str], threshold: float = 0.5) -> List[Dict]:
        """
        Make predictions on new texts
        
        Args:
            texts: List of texts to classify
            threshold: Threshold for binary classification
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get model output
                outputs = self.model(**encoding)
                logits = outputs['logits']
                
                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
                
                # Binary predictions based on threshold
                binary_predictions = (probabilities > threshold).astype(int)
                
                # Get predicted labels
                predicted_labels = [
                    self.label_encoder.classes_[i] 
                    for i, pred in enumerate(binary_predictions) if pred == 1
                ]
                
                # Create probability dictionary
                prob_dict = {
                    self.label_encoder.classes_[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                }
                
                predictions.append({
                    'text': text,
                    'predicted_labels': predicted_labels,
                    'probabilities': prob_dict,
                    'binary_predictions': binary_predictions,
                    'max_probability': float(np.max(probabilities)),
                    'confidence': float(np.max(probabilities)) if predicted_labels else 0.0
                })
                
        return predictions
    
    def load_model(self, model_path: str):
        """
        Load a pre-trained model
        
        Args:
            model_path: Path to the saved model directory
        """
        logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load label encoder
        with open(f'{model_path}/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        # Load model
        num_labels = len(self.label_encoder.classes_)
        self.model = BioBERTClassifier(
            model_name=self.model_name,
            num_labels=num_labels
        )
        
        # Load state dict
        state_dict = torch.load(f'{model_path}/pytorch_model.bin', map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        
        logger.info("Model loaded successfully!")
    
    def get_feature_importance(self, text: str, predicted_class: str = None) -> Dict:
        """
        Get attention weights as feature importance
        
        Args:
            text: Input text
            predicted_class: Specific class to analyze (optional)
            
        Returns:
            Dictionary with attention weights and important tokens
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        self.model.eval()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            attentions = outputs['attentions']
            
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        # Average attention across heads and layers (use last layer)
        attention_weights = attentions[-1][0].mean(dim=0).cpu().numpy()
        
        # Get attention for [CLS] token (first token)
        cls_attention = attention_weights[0, :]
        
        # Create importance scores
        token_importance = []
        for i, (token, score) in enumerate(zip(tokens, cls_attention)):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                token_importance.append({
                    'token': token,
                    'importance': float(score),
                    'position': i
                })
        
        # Sort by importance
        token_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'text': text,
            'token_importance': token_importance[:20],  # Top 20 tokens
            'attention_matrix': attention_weights.tolist()
        }
    
    def save_results(self, output_path: str):
        """
        Save evaluation results and training history
        
        Args:
            output_path: Path to save results JSON file
        """
        results = {
            'model_info': {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'device': self.device,
                'num_labels': len(self.label_encoder.classes_) if self.label_encoder else 0,
                'label_classes': self.label_encoder.classes_.tolist() if self.label_encoder else []
            },
            'training_history': self.training_history,
            'evaluation_results': self.evaluation_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {output_path}")

# Example usage and testing functions
def create_sample_medical_data() -> pd.DataFrame:
    """
    Create sample medical data for testing
    """
    sample_data = [
        {
            'title': 'Neurological manifestations of COVID-19',
            'abstract': 'This study examines the neurological complications associated with COVID-19 infection, including stroke, seizures, and cognitive impairment.',
            'domains': ['neurological']
        },
        {
            'title': 'Cardiovascular risk factors in diabetes',
            'abstract': 'Analysis of cardiovascular complications in diabetic patients, focusing on hypertension, coronary artery disease, and heart failure.',
            'domains': ['cardiovascular']
        },
        {
            'title': 'Hepatocellular carcinoma treatment outcomes',
            'abstract': 'Evaluation of treatment modalities for hepatocellular carcinoma including surgical resection, transplantation, and targeted therapy.',
            'domains': ['hepatorenal', 'oncological']
        },
        {
            'title': 'Brain tumor classification using MRI',
            'abstract': 'Machine learning approaches for classifying brain tumors from MRI images, with focus on glioblastoma and meningioma detection.',
            'domains': ['neurological', 'oncological']
        }
    ]
    
    df = pd.DataFrame(sample_data)
    df['combined_text'] = df['title'] + ' ' + df['abstract']
    
    return df

if __name__ == "__main__":
    # Example usage
    print("BioBERT Medical Classifier - Example Usage")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_medical_data()
    print(f"Created sample dataset with {len(df)} samples")
    
    # Initialize classifier
    classifier = BioBERTMedicalClassifier()
    
    # Note: For actual training, you would need a larger dataset
    print("\nTo train the model:")
    print("1. Prepare your CSV file with 'title', 'abstract', and 'domains' columns")
    print("2. Run: classifier.train(train_df, val_df)")
    print("3. Evaluate: classifier.evaluate(test_df)")
    print("4. Make predictions: classifier.predict(['Your medical text here'])")
    
    print(f"\nModel will use: {classifier.device}")
    print(f"BioBERT model: {classifier.model_name}")
