"""Busca leve de hiperparâmetros usando treino/validação (sem cruzamento com teste).

Escolhe a configuração que maximiza F1-macro na validação entre um grid pequeno.
Salva modelo/vetorizador em `modelos/` e resultados em `resultados/`.
"""
import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from stopwords_pt import STOPWORDS_PT
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from dividir_dados import sha256_arquivo, validar_particoes


RAIZ = Path(__file__).resolve().parents[1]
PASTA_RESULT = RAIZ / 'resultados'
PASTA_MODELOS = RAIZ / 'modelos'
PASTA_RESULT.mkdir(exist_ok=True)
PASTA_MODELOS.mkdir(exist_ok=True)


def avaliar_config(treino, validacao, tfidf_kwargs, log_kwargs):
    vet = TfidfVectorizer(**tfidf_kwargs)
    x_tr = vet.fit_transform(treino.texto_tfidf)
    x_val = vet.transform(validacao.texto_tfidf)
    modelo = LogisticRegression(**log_kwargs)
    modelo.fit(x_tr, treino.rotulo)
    p = modelo.predict(x_val)
    return float(f1_score(validacao.rotulo, p, average='macro', labels=[0, 1])), vet, modelo


def main():
    caminhos = {n: RAIZ / 'dados_processados' / f'{n}.csv' for n in ['treino', 'validacao']}
    partes = {n: pd.read_csv(p, keep_default_na=False) for n, p in caminhos.items()}
    validar_particoes({**partes, 'teste': pd.read_csv(RAIZ / 'dados_processados/teste.csv', keep_default_na=False)})
    treino, validacao = partes['treino'], partes['validacao']

    grid_tfidf = [
        {'ngram_range': (1, 1), 'min_df': 3, 'max_features': 50000, 'sublinear_tf': True, 'strip_accents': 'unicode', 'stop_words': STOPWORDS_PT},
        {'ngram_range': (1, 2), 'min_df': 3, 'max_features': 50000, 'sublinear_tf': True, 'strip_accents': 'unicode', 'stop_words': STOPWORDS_PT},
        {'ngram_range': (1, 2), 'min_df': 5, 'max_features': 50000, 'sublinear_tf': True, 'strip_accents': 'unicode', 'stop_words': STOPWORDS_PT},
    ]
    grid_log = [
        {'C': 0.1, 'max_iter': 1000, 'solver': 'liblinear', 'random_state': 42},
        {'C': 1.0, 'max_iter': 1000, 'solver': 'liblinear', 'random_state': 42},
        {'C': 10.0, 'max_iter': 1000, 'solver': 'liblinear', 'random_state': 42},
    ]

    best = {'f1': -1}
    resultados = []
    for tfidf_kws, log_kws in product(grid_tfidf, grid_log):
        nome = f"ng{tfidf_kws['ngram_range']}_md{tfidf_kws['min_df']}_C{log_kws['C']}"
        try:
            f1, vet, modelo = avaliar_config(treino, validacao, tfidf_kws, log_kws)
        except Exception as e:
            resultados.append({'nome': nome, 'erro': str(e)})
            continue
        resultados.append({'nome': nome, 'f1_macro': f1, 'tfidf': tfidf_kws, 'logistica': log_kws})
        if f1 > best['f1']:
            best = {'f1': f1, 'nome': nome, 'tfidf': tfidf_kws, 'logistica': log_kws, 'vetorizador': vet, 'modelo': modelo}

    # Salvar resultados e melhor modelo
    (PASTA_RESULT / 'busca_resultados.json').write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding='utf-8')
    resumo = {k: v for k, v in best.items() if k not in ('vetorizador', 'modelo')}
    (PASTA_RESULT / 'busca_melhor.json').write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding='utf-8')
    if 'modelo' in best:
        joblib.dump(best['modelo'], PASTA_MODELOS / 'modelo_tuned.joblib')
        joblib.dump(best['vetorizador'], PASTA_MODELOS / 'vetorizador_tuned.joblib')
    print('Busca concluída. Melhor f1_macro na validação:', best.get('f1'))


if __name__ == '__main__':
    main()
