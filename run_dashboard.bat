@echo off
cd /d "%~dp0"
echo Starting NAP Legal AI Advisor from: %CD%
streamlit run nap-legal-ai-advisor/app.py %*
