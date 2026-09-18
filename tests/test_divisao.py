"""Contratos da divisão: testes sintéticos e auditoria dos CSVs reais."""
from pathlib import Path

import pandas as pd
import pytest

from src.preparar_divisao import preparar_base, texto_seguro
from src.dividir_dados import dividir_base, validar_particoes

RAIZ = Path(__file__).resolve().parents[1]


def base_sintetica():
    linhas = []
    for evento in range(120):
        for rotulo in (0, 1):
            linhas.append(dict(id=f'fakebr_{rotulo}_{evento}',
                id_evento=f'fakebr_evento_{evento}', dataset_origem='Fake.Br',
                rotulo=rotulo, texto_original=f'Notícia {evento} versão {rotulo} não é igual.',
                texto_bert=f'Notícia {evento} versão {rotulo} não é igual.', titulo=''))
    for evento in range(240):
        linhas.append(dict(id=f'rec_{evento}', id_evento=f'rec_{evento}',
            dataset_origem='FakeRecogna', rotulo=evento % 2,
            texto_original=f'Outro relato {evento} nunca sem informação.',
            texto_bert=f'Outro relato {evento} nunca sem informação.', titulo=''))
    return pd.DataFrame(linhas)


@pytest.fixture(scope='module')
def sintetica():
    df, _, _ = preparar_base(base_sintetica())
    return df, dividir_base(df)[0]


def test_higiene_preserva_negacoes_e_acentos():
    texto = texto_seguro('Boato – Não, nunca, sem, jamais! ação 2024 #boato #boatos '
                         'https://exemplo.com autor@site.com @autor Boatos.org')
    assert all(p in texto for p in ['não', 'nunca', 'sem', 'jamais', 'ação', '2024'])
    assert not any(p in texto for p in ['boato', 'http', '@', '#', 'site.com'])


def test_corpo_nao_inclui_titulo_ou_metadados():
    original = base_sintetica()
    original['titulo'] = [f'Segredo editorial exclusivo {i}' for i in range(len(original))]
    df, _, _ = preparar_base(original)
    assert not df.texto_tfidf.str.contains('segredo|editorial').any()
    assert df.set_index('id').texto_original.equals(original.set_index('id').texto_original)
    assert df.set_index('id').texto_bert.equals(original.set_index('id').texto_bert)


def test_duplicatas_unem_eventos_antes_de_remover():
    original = base_sintetica()
    # Dois pares Fake.Br ligados por um corpo repetido: seus parceiros devem ficar juntos.
    original.loc[2, 'texto_original'] = original.loc[0, 'texto_original']
    df, auditoria, removidos = preparar_base(original)
    assert len(df) == len(original) - 1
    assert df.loc[df.id.isin(original.loc[:3, 'id']), 'id_evento'].nunique() == 1
    assert len(removidos) == 1
    assert auditoria['removidos'] == 1


def test_conflitos_removem_todas_as_copias():
    original = base_sintetica()
    original.loc[1, 'texto_original'] = original.loc[0, 'texto_original']
    df, _, removidos = preparar_base(original)
    assert not set(original.loc[:1, 'id']) & set(df.id)
    assert set(removidos.motivo) == {'rotulos_conflitantes'}


def test_titulos_identicos_agrupados_sem_apagar_noticias():
    original = base_sintetica()
    original.loc[0, 'titulo'] = 'A mesma alegação suficientemente longa #boato'
    original.loc[2, 'titulo'] = 'A mesma alegação suficientemente longa'
    df, _, _ = preparar_base(original)
    assert len(df) == len(original)
    assert df.loc[df.id.isin(original.loc[:3, 'id']), 'id_evento'].nunique() == 1


def test_grupos_reprodutiveis_e_balanceados(sintetica):
    df, partes = sintetica
    repetir, _ = dividir_base(df)
    validar_particoes(partes)
    for nome, alvo in [('treino', .7), ('validacao', .15), ('teste', .15)]:
        assert partes[nome].id.tolist() == repetir[nome].id.tolist()
        assert abs(len(partes[nome]) / len(df) - alvo) <= .02
        assert abs(partes[nome].rotulo.mean() - df.rotulo.mean()) <= .03
    assert set(pd.concat(partes.values()).id) == set(df.id)


def test_vazamento_rejeitado(sintetica):
    _, partes = sintetica
    adulteradas = {k: v.copy() for k, v in partes.items()}
    adulteradas['teste'].loc[adulteradas['teste'].index[0], 'id_evento'] = partes['treino'].iloc[0].id_evento
    with pytest.raises(ValueError, match='evento'):
        validar_particoes(adulteradas)


@pytest.mark.parametrize('campo,valor', [('id_evento', ''), ('rotulo', 3), ('id', '')])
def test_entrada_invalida_rejeitada(campo, valor):
    df = base_sintetica()
    df.loc[0, campo] = valor
    with pytest.raises(ValueError):
        preparar_base(df)


@pytest.fixture(scope='module')
def reais():
    caminhos = {n: RAIZ / 'dados_processados' / f'{n}.csv' for n in ['treino', 'validacao', 'teste']}
    assert all(p.exists() for p in caminhos.values()), 'Execute src/dividir_dados.py antes de pytest.'
    return {n: pd.read_csv(p, keep_default_na=False) for n, p in caminhos.items()}


def test_reais_sem_intersecao_e_com_duas_classes(reais):
    validar_particoes(reais)
    nomes = list(reais)
    for i, nome in enumerate(nomes):
        df = reais[nome]
        assert set(df.rotulo) == {0, 1}
        for coluna in ['texto_original', 'texto_bert', 'texto_tfidf']:
            assert df[coluna].str.strip().ne('').all()
        for outro in nomes[i + 1:]:
            assert not set(df.id_evento) & set(reais[outro].id_evento)
            assert not set(df.id_evento_original) & set(reais[outro].id_evento_original)


def test_reais_duplicatas_proporcoes_e_marcadores(reais):
    tudo = pd.concat(reais.values())
    for campo in ['id', 'texto_original', 'texto_tfidf', 'chave_modelo']:
        assert not tudo[campo].duplicated().any()
    assert not tudo.texto_tfidf.str.contains(r'#|https?://|www\.|@|boatos?\.org', case=False).any()
    for nome, alvo in [('treino', .7), ('validacao', .15), ('teste', .15)]:
        df = reais[nome]
        assert abs(len(df) / len(tudo) - alvo) <= .02
        assert abs(df.rotulo.mean() - tudo.rotulo.mean()) <= .03
        for origem in tudo.dataset_origem.unique():
            sub = df[df.dataset_origem.eq(origem)]
            assert set(sub.rotulo) == {0, 1}
            assert abs(sub.rotulo.mean() - tudo[tudo.dataset_origem.eq(origem)].rotulo.mean()) <= .03


def test_reais_pares_fakebr_e_cobertura(reais):
    original = pd.read_csv(RAIZ / 'dados_processados/noticias_limpas.csv', keep_default_na=False)
    tudo = pd.concat([df.assign(conjunto=n) for n, df in reais.items()])
    fakebr = tudo[tudo.dataset_origem.eq('Fake.Br')]
    numeros = fakebr.id.str.rsplit('_', n=1).str[-1]
    assert fakebr.groupby(numeros).conjunto.nunique().eq(1).all()
    removidos = pd.read_csv(RAIZ / 'resultados/registros_excluidos_divisao.csv', keep_default_na=False)
    assert set(tudo.id).isdisjoint(removidos.id)
    assert set(tudo.id) | set(removidos.id) == set(original.id)
    # Cada par original completo está preservado, ou sua exclusão foi explicitamente auditada.
    destinos = tudo.set_index('id').conjunto.to_dict()
    excluidos = set(removidos.id)
    for _, par in original[original.dataset_origem.eq('Fake.Br')].groupby('id_evento'):
        assert all(i in destinos or i in excluidos for i in par.id)
        assert len({destinos[i] for i in par.id if i in destinos}) <= 1
    comparar = tudo.set_index('id')
    origem = original.set_index('id').loc[comparar.index]
    assert comparar.texto_original.equals(origem.texto_original)
    assert comparar.texto_bert.equals(origem.texto_bert)
