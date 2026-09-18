"""CLI simples para inferência em lote usando o modelo salvo.

Exemplos:
  # Inferir a partir de um CSV com coluna `texto` e salvar previsões
  python3 src/inferencia_cli.py --input dados.csv --output previsoes.csv

  # Inferir um único texto
  python3 src/inferencia_cli.py --text "Exemplo de notícia" 
"""
import argparse
from pathlib import Path
import sys
import csv

import joblib
import pandas as pd


def carregar_modelo(vet_path, model_path):
    vet = joblib.load(vet_path)
    model = joblib.load(model_path)
    return vet, model


def inferir_dataframe(df, vet, model, threshold=0.5):
    if 'texto' not in df.columns and 'texto_original' in df.columns:
        df = df.rename(columns={'texto_original': 'texto'})
    textos = df['texto'].fillna('').astype(str)
    X = vet.transform(textos)
    probs = model.predict_proba(X)[:, list(model.classes_).index(1)]
    preds = (probs >= threshold).astype(int)
    out = df.copy()
    out['probabilidade_verdadeiro'] = probs
    out['rotulo_previsto'] = preds
    return out


def inferir_texto(texto, vet, model, threshold=0.5):
    df = pd.DataFrame({'texto': [texto]})
    return inferir_dataframe(df, vet, model, threshold)


def main():
    p = argparse.ArgumentParser(description='Inferência em lote com modelo TF-IDF + logística')
    p.add_argument('--input', '-i', type=Path, help='CSV de entrada com coluna `texto`')
    p.add_argument('--output', '-o', type=Path, default=Path('previsoes_output.csv'), help='CSV de saída')
    p.add_argument('--model', type=Path, default=Path('modelos/modelo_tfidf_logistica.joblib'), help='Caminho do modelo joblib')
    p.add_argument('--vet', type=Path, default=Path('modelos/vetorizador_tfidf.joblib'), help='Caminho do vetorizador joblib')
    p.add_argument('--text', type=str, help='Inferir um único texto (ignora --input)')
    p.add_argument('--threshold', type=float, default=0.5, help='Limiar para rotulo verdadeiro')
    args = p.parse_args()

    if not args.input and not args.text:
        p.print_usage()
        print('\nInforme --input CSV ou --text "texto"')
        sys.exit(1)

    vet, model = carregar_modelo(args.vet, args.model)

    if args.text:
        out = inferir_texto(args.text, vet, model, args.threshold)
    else:
        df = pd.read_csv(args.input, keep_default_na=False)
        out = inferir_dataframe(df, vet, model, args.threshold)

    out.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'Previsões salvas em {args.output}')


if __name__ == '__main__':
    main()
