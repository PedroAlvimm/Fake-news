"""Baseline TF-IDF + regressão logística, com avaliação final protegida."""
import json
import os
import platform
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

if __package__:
    from .dividir_dados import sha256_arquivo, validar_particoes
else:
    from dividir_dados import sha256_arquivo, validar_particoes

if __package__:
    from .stopwords_pt import STOPWORDS_PT
else:
    from stopwords_pt import STOPWORDS_PT

RAIZ = Path(__file__).resolve().parents[1]
CONFIGURACAO = {
    'entrada': 'texto_tfidf', 'random_state': 42,
    'tfidf': {'ngram_range': (1, 2), 'min_df': 3, 'max_features': 50000,
              'sublinear_tf': True, 'strip_accents': 'unicode', 'stop_words': STOPWORDS_PT},
    'logistica': {'C': 1.0, 'max_iter': 1000, 'solver': 'liblinear', 'random_state': 42},
    'limiar_verdadeiro': .5, 'razao_minima_para_pesos': 1.5,
}


def escolher_peso(rotulos):
    contagens = pd.Series(rotulos).value_counts()
    if set(contagens.index) != {0, 1}:
        raise ValueError('Treinamento requer as duas classes.')
    return 'balanced' if contagens.max() / contagens.min() >= 1.5 else None


def ajustar_modelo(treino):
    """Nenhuma outra coluna do DataFrame entra na matriz de atributos."""
    vetorizador = TfidfVectorizer(**CONFIGURACAO['tfidf'])
    x = vetorizador.fit_transform(treino['texto_tfidf'])
    modelo = LogisticRegression(**CONFIGURACAO['logistica'], class_weight=escolher_peso(treino.rotulo))
    # Não seguir silenciosamente com um modelo que não convergiu.
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        modelo.fit(x, treino.rotulo)
    return vetorizador, modelo


def calcular_metricas(real, previsto, probabilidade_verdadeiro):
    real, previsto = np.asarray(real), np.asarray(previsto)
    p = np.asarray(probabilidade_verdadeiro)
    resultado = {'n': len(real), 'acuracia': float(accuracy_score(real, previsto)),
                 'f1_macro': float(f1_score(real, previsto, average='macro', labels=[0, 1], zero_division=0)),
                 'ordem_classes_matriz': ['falso (0)', 'verdadeiro (1)'],
                 'matriz_confusao': confusion_matrix(real, previsto, labels=[0, 1]).tolist()}
    for classe, nome in [(0, 'falso'), (1, 'verdadeiro')]:
        resultado[nome] = {
            'precisao': float(precision_score(real, previsto, pos_label=classe, zero_division=0)),
            'recall': float(recall_score(real, previsto, pos_label=classe, zero_division=0)),
            'f1': float(f1_score(real, previsto, pos_label=classe, zero_division=0)),
            'suporte': int((real == classe).sum()),
        }
    duas_classes = len(np.unique(real)) == 2
    resultado['roc_auc_verdadeiro'] = float(roc_auc_score(real, p)) if duas_classes else None
    resultado['roc_auc_falso'] = float(roc_auc_score(real == 0, 1 - p)) if duas_classes else None
    resultado['relatorio_classificacao'] = classification_report(
        real, previsto, labels=[0, 1], target_names=['falso', 'verdadeiro'], output_dict=True, zero_division=0)
    return resultado


def prever(dados, vetorizador, modelo):
    probabilidades = modelo.predict_proba(vetorizador.transform(dados.texto_tfidf))
    indice_verdadeiro = list(modelo.classes_).index(1)
    p = probabilidades[:, indice_verdadeiro]
    previsto = (p >= CONFIGURACAO['limiar_verdadeiro']).astype(int)
    return pd.DataFrame({
        'id': dados.id.to_numpy(), 'dataset_origem': dados.dataset_origem.to_numpy(),
        'rotulo_real': dados.rotulo.to_numpy(), 'rotulo_previsto': previsto,
        'probabilidade_verdadeiro': p, 'probabilidade_falso': 1 - p,
        'acerto': previsto == dados.rotulo.to_numpy(),
    })


def avaliar(previsoes):
    subconjuntos = {'geral': previsoes}
    subconjuntos.update({origem: parte for origem, parte in previsoes.groupby('dataset_origem')})
    metricas = {}
    for nome, parte in subconjuntos.items():
        metricas[nome] = calcular_metricas(parte.rotulo_real, parte.rotulo_previsto, parte.probabilidade_verdadeiro)
        print(f'\n{nome} — n={len(parte)}')
        print(f"Acurácia: {metricas[nome]['acuracia']:.4f}; F1 macro: {metricas[nome]['f1_macro']:.4f}; "
              f"ROC-AUC: {metricas[nome]['roc_auc_verdadeiro']}")
        print(classification_report(parte.rotulo_real, parte.rotulo_previsto, labels=[0, 1],
                                    target_names=['falso (interesse=0)', 'verdadeiro (1)'], zero_division=0, digits=4))
        print('Matriz (linhas=reais, colunas=previstas; ordem 0, 1):', metricas[nome]['matriz_confusao'])
    return metricas


def salvar_relatorios(metricas, etapa, pasta):
    # Backend sem interface gráfica, com cache em diretório gravável do projeto.
    os.environ.setdefault('MPLCONFIGDIR', str(pasta / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    linhas = []
    for origem, m in metricas.items():
        for classe, valores in m['relatorio_classificacao'].items():
            if isinstance(valores, dict):
                linhas.append({'etapa': etapa, 'dataset_origem': origem, 'classe': classe, **valores})
        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        ConfusionMatrixDisplay(np.array(m['matriz_confusao']), display_labels=['Falso (0)', 'Verdadeiro (1)']).plot(
            ax=ax, cmap='Blues', colorbar=False, values_format='d')
        ax.set(title=f'{etapa.capitalize()} — {origem}', xlabel='Classe prevista', ylabel='Classe real')
        fig.tight_layout()
        nome = origem.lower().replace('.', '')
        fig.savefig(pasta / f'matriz_confusao_{etapa}_{nome}.png', dpi=160)
        plt.close(fig)
    pd.DataFrame(linhas).to_csv(pasta / f'relatorio_classificacao_{etapa}.csv', index=False, encoding='utf-8-sig')


def salvar_json(caminho, conteudo):
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    pasta = RAIZ / 'resultados'
    modelos = RAIZ / 'modelos'
    pasta.mkdir(exist_ok=True)
    modelos.mkdir(exist_ok=True)
    registro = pasta / 'avaliacao_final.json'
    if registro.exists() or (pasta / 'metricas.json').exists() or (modelos / 'modelo_tfidf_logistica.joblib').exists():
        raise FileExistsError('Experimento já treinado/iniciado. Consulte resultados; não reutilize o teste para ajustes.')
    manifesto = json.loads((pasta / 'divisao.json').read_text(encoding='utf-8'))
    caminhos = {n: RAIZ / 'dados_processados' / f'{n}.csv' for n in ['treino', 'validacao', 'teste']}
    hashes = {n: sha256_arquivo(p) for n, p in caminhos.items()}
    if any(hashes[n] != manifesto['conjuntos'][n]['sha256'] for n in caminhos):
        raise ValueError('CSV alterado após a divisão; hashes não correspondem ao manifesto.')
    # Ler o teste aqui serve apenas à auditoria estrutural. Nenhum fit depende dele.
    partes = {n: pd.read_csv(p, keep_default_na=False) for n, p in caminhos.items()}
    validar_particoes(partes)
    treino, validacao, teste = (partes[n] for n in ['treino', 'validacao', 'teste'])
    vetorizador, modelo = ajustar_modelo(treino)
    print('\nVALIDAÇÃO — configuração fixa, sem busca de hiperparâmetros')
    metricas_validacao = avaliar(prever(validacao, vetorizador, modelo))
    majoritaria = int(treino.rotulo.value_counts().idxmax())
    acuracia_majoritaria = float(validacao.rotulo.eq(majoritaria).mean())
    configuracao = {**CONFIGURACAO, 'class_weight': modelo.class_weight,
        'atributos_aprendidos': len(vetorizador.vocabulary_), 'iteracoes': modelo.n_iter_.tolist(),
        'acuracia_majoritaria_validacao': acuracia_majoritaria,
        'decisao_validacao': 'Prosseguir somente se acurácia superar predição constante da classe majoritária; sem ajustes.',
        'sha256_conjuntos': hashes,
        'versoes': {p: version(p) for p in ['scikit-learn', 'pandas', 'numpy', 'joblib', 'matplotlib']},
        'python': platform.python_version()}
    salvar_json(pasta / 'metricas_validacao.json', {'configuracao': configuracao, 'validacao': metricas_validacao})
    salvar_relatorios(metricas_validacao, 'validacao', pasta)
    if metricas_validacao['geral']['acuracia'] <= acuracia_majoritaria:
        raise ValueError('Validação não superou baseline majoritário; teste não foi avaliado.')
    # Criação exclusiva evita duas avaliações concorrentes ou repetidas acidentalmente.
    with registro.open('x', encoding='utf-8') as arquivo:
        json.dump({'estado': 'iniciada', 'utc': datetime.now(timezone.utc).isoformat(),
                   'configuracao_congelada': configuracao}, arquivo, ensure_ascii=False, indent=2)
    joblib.dump(modelo, modelos / 'modelo_tfidf_logistica.joblib')
    joblib.dump(vetorizador, modelos / 'vetorizador_tfidf.joblib')
    print('\nTESTE FINAL — uma única chamada predict_proba; subconjuntos reutilizam suas previsões')
    previsoes = prever(teste, vetorizador, modelo)
    previsoes.to_csv(pasta / 'previsoes_teste.csv', index=False, encoding='utf-8-sig')
    metricas_teste = avaliar(previsoes)
    salvar_json(pasta / 'metricas.json', {'configuracao': configuracao, 'validacao': metricas_validacao, 'teste': metricas_teste})
    salvar_relatorios(metricas_teste, 'teste', pasta)
    registro_final = json.loads(registro.read_text(encoding='utf-8'))
    registro_final.update(estado='concluida', sha256_previsoes=sha256_arquivo(pasta / 'previsoes_teste.csv'))
    salvar_json(registro, registro_final)
    print('\nModelo, vetorizador, métricas, matrizes e previsões salvos.')


if __name__ == '__main__':
    main()
