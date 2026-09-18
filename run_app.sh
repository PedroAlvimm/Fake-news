#!/usr/bin/env bash
set -euo pipefail

# Script para ativar o venv (ou criar) e rodar o app Streamlit
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
  echo "Ativando ambiente virtual em $VENV_DIR"
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
else
  echo "Ambiente virtual não encontrado. Criando $VENV_DIR e instalando dependências..."
  python3 -m venv "$VENV_DIR"
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements.txt
fi

echo "Iniciando Streamlit (http://localhost:8501)"
streamlit run src/app_streamlit.py
