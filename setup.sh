#!/bin/bash

echo "🚀 Setting up GreyOak Dashboard for Streamlit Cloud..."

# Create directories
mkdir -p .streamlit
mkdir -p greyoak_v4_production/daily_signals
mkdir -p greyoak_v4_production/summary
mkdir -p greyoak_v4_production/debug
mkdir -p greyoak_v4_production/strong_signals
mkdir -p greyoak_v4_production/top_signals
mkdir -p compressed_outputs

# Create config
cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 500
enableXsrfProtection = true
enableCORS = false
maxMessageSize = 500

[browser]
gatherUsageStats = false
EOF

# Create requirements
cat > requirements.txt << 'EOF'
streamlit==1.28.0
pandas>=2.1.0
numpy>=1.24.0
plotly>=5.17.0
yfinance>=0.2.28
openpyxl>=3.1.2
tqdm>=4.65.0
requests>=2.31.0
EOF

echo "✅ Setup complete!"
echo ""
echo "📋 To deploy:"
echo "1. Push to GitHub"
echo "2. Go to https://share.streamlit.io"
echo "3. Deploy with main file: streamlit_app.py"
echo ""
echo "🌐 Your dashboard will be live at:"
echo "   https://your-username-greyoak-dashboard.streamlit.app"
