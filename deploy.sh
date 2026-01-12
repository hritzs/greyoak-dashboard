#!/bin/bash

# GreyOak SlopeTrigger v4.1 - Streamlit Deployment Script

echo "🚀 Deploying GreyOak Dashboard..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p ./greyoak_v4_production/daily_signals
mkdir -p ./greyoak_v4_production/summary
mkdir -p ./greyoak_v4_production/debug
mkdir -p ./greyoak_v4_production/strong_signals
mkdir -p ./greyoak_v4_production/top_signals
mkdir -p ./compressed_outputs

echo "✅ Setup complete!"
echo ""
echo "📋 To run locally:"
echo "   streamlit run greyoak_streamlit.py"
echo ""
echo "📋 To deploy on Streamlit Cloud:"
echo "   1. Push to GitHub"
echo "   2. Go to https://share.streamlit.io"
echo "   3. Deploy with main file: greyoak_streamlit.py"
echo ""
echo "🌐 Your dashboard will be live at:"
echo "   https://your-username-greyoak-dashboard.streamlit.app"