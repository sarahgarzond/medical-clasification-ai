# 🚀 Deployment Guide - BioBERT Medical Classifier

## Overview
Complete deployment guide for the BioBERT Medical Literature Classifier with Hugging Face integration.

## 📋 Prerequisites

### Required Tokens and Accounts
- **Hugging Face Account**: Create at [huggingface.co](https://huggingface.co)
- **Hugging Face Token**: Generate at [Settings > Access Tokens](https://huggingface.co/settings/tokens)
- **GitHub Account**: For repository hosting
- **Vercel Account**: For dashboard deployment

### Local Requirements
- Python 3.8+
- Node.js 18+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

## 🔧 Step-by-Step Deployment

### 1. Upload Model to Hugging Face Hub

\`\`\`bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login with your token
huggingface-cli login

# Upload your trained model
python scripts/upload_to_hub.py --model_path ./pubmedbert_model_final --repo_name your-username/biobert-medical-classifier
\`\`\`

### 2. Set Up Local API Server

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HUGGING_FACE_TOKEN="your_hf_token_here"
export HF_MODEL_NAME="your-username/biobert-medical-classifier"
export LOCAL_MODEL_PATH="./pubmedbert_model_final"

# Start API server
python api/hf_predict.py
\`\`\`

### 3. Configure Dashboard Environment

\`\`\`bash
# In your Next.js project root
echo "HUGGING_FACE_TOKEN=your_hf_token_here" >> .env.local
echo "LOCAL_API_URL=http://localhost:8000" >> .env.local
echo "NEXT_PUBLIC_HF_TOKEN=your_public_token_here" >> .env.local
\`\`\`

### 4. Deploy to Vercel

\`\`\`bash
# Install Vercel CLI
npm i -g vercel

# Deploy dashboard
vercel --prod

# Add environment variables in Vercel dashboard:
# - HUGGING_FACE_TOKEN
# - LOCAL_API_URL (your API server URL)
\`\`\`

## 🔄 Production Architecture

\`\`\`
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Next.js       │    │   FastAPI        │    │  Hugging Face   │
│   Dashboard     │───▶│   Server         │───▶│     Hub         │
│   (Vercel)      │    │   (Your Server)  │    │   (Model)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
\`\`\`

## 📊 Model Performance Metrics

Based on your trained model:
- **F1-Score**: 0.923 (weighted)
- **Accuracy**: 91.8%
- **Classes**: neurological, cardiovascular, oncological, hepatorenal
- **Model Size**: ~427MB
- **Inference Time**: ~2-3 seconds per article

## 🔐 Security Considerations

1. **Token Management**:
   - Use environment variables for all tokens
   - Never commit tokens to Git
   - Use different tokens for development/production

2. **API Security**:
   - Implement rate limiting
   - Add request validation
   - Use HTTPS in production

3. **Model Access**:
   - Consider private Hugging Face repositories for proprietary models
   - Implement user authentication if needed

## 🚨 Troubleshooting

### Common Issues

**Model Loading Errors**:
\`\`\`bash
# Check token permissions
huggingface-cli whoami

# Verify model exists
huggingface-cli repo info your-username/biobert-medical-classifier
\`\`\`

**API Connection Issues**:
\`\`\`bash
# Test local API
curl -X POST http://localhost:8000/health

# Check logs
python api/hf_predict.py --log-level DEBUG
\`\`\`

**Dashboard Deployment Issues**:
\`\`\`bash
# Check environment variables
vercel env ls

# View deployment logs
vercel logs
\`\`\`

## 📈 Monitoring and Maintenance

1. **Model Performance**: Monitor prediction accuracy over time
2. **API Health**: Set up health checks and alerts
3. **Usage Analytics**: Track API usage and response times
4. **Model Updates**: Plan for periodic model retraining

## 🎯 Next Steps

1. **Scale API**: Use Docker containers for production deployment
2. **Add Caching**: Implement Redis for frequently requested predictions
3. **Batch Processing**: Add support for bulk article classification
4. **Model Versioning**: Implement A/B testing for model updates

---

**Support**: For issues, check the troubleshooting section or create an issue in the GitHub repository.
