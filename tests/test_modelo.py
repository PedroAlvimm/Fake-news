"""Métricas, isolamento do vocabulário e interpretação inequívoca dos erros."""
import numpy as np
import pandas as pd
import pytest

from src.treinar_modelo_base import calcular_metricas, ajustar_modelo, escolher_peso
from src.analisar_erros import montar_erros


def test_metricas_classe_falsa_e_auc():
    y = np.array([0, 0, 0, 1, 1])
    pred = np.array([0, 1, 0, 0, 1])
    prob = np.array([.1, .7, .2, .3, .9])
    m = calcular_metricas(y, pred, prob)
    assert m['acuracia'] == pytest.approx(.6)
    assert m['falso']['recall'] == pytest.approx(2/3)
    assert m['falso']['precisao'] == pytest.approx(2/3)
    assert m['verdadeiro']['recall'] == pytest.approx(.5)
    assert m['matriz_confusao'] == [[2, 1], [1, 1]]
    assert m['roc_auc_verdadeiro'] == pytest.approx(5/6)
    assert m['roc_auc_falso'] == pytest.approx(5/6)


def test_auc_indefinida_classe_unica():
    m = calcular_metricas([0, 0], [0, 1], [.2, .8])
    assert m['roc_auc_verdadeiro'] is None
    assert m['roc_auc_falso'] is None
    assert m['matriz_confusao'] == [[1, 1], [0, 0]]


def test_peso_apenas_quando_desbalanceado():
    assert escolher_peso([0, 0, 1, 1]) is None
    assert escolher_peso([0, 0, 0, 1]) == 'balanced'


def test_vocabulario_aprendido_so_no_treino():
    treino = pd.DataFrame({'texto_tfidf': ['alegação nunca confirmada'] * 4 + ['notícia confirmada hoje'] * 4,
                          'rotulo': [0]*4 + [1]*4, 'url': ['segredometadado'] * 8})
    vetorizador, modelo = ajustar_modelo(treino)
    vetorizador.transform(['exclusivodavalidacao nunca confirmado'])
    assert 'exclusivodavalidacao' not in vetorizador.vocabulary_
    assert 'segredometadado' not in vetorizador.vocabulary_
    assert 'nunca' in vetorizador.vocabulary_
    assert set(modelo.classes_) == {0, 1}


def test_erros_significados_e_ordem():
    teste = pd.DataFrame({'id': ['a', 'b', 'c'], 'rotulo': [0, 1, 0],
        'dataset_origem': ['Fake.Br']*3, 'texto_original': ['exemplo A', 'exemplo B', 'exemplo C'],
        'categoria': ['', 'saúde', '']})
    previsoes = pd.DataFrame({'id': ['a', 'b', 'c'], 'dataset_origem': ['Fake.Br']*3,
        'rotulo_real': [0, 1, 0], 'rotulo_previsto': [1, 0, 0],
        'probabilidade_verdadeiro': [.8, .1, .2], 'probabilidade_falso': [.2, .9, .8],
        'acerto': [False, False, True]})
    erros = montar_erros(previsoes, teste)
    assert erros.id.tolist() == ['b', 'a']
    assert erros.tipo_erro.tolist() == ['verdadeira classificada como fake', 'fake classificada como verdadeira']
    assert erros.confianca.tolist() == [.9, .8]
    assert erros.categoria.tolist() == ['saúde', 'não informada']
    previsoes.loc[0, 'rotulo_real'] = 1
    with pytest.raises(ValueError, match='rótulos'):
        montar_erros(previsoes, teste)
