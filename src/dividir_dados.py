"""Divisão reproduzível por eventos, com busca limitada de balanceamento."""
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

if __package__:
    from .preparar_divisao import preparar_base
else:
    from preparar_divisao import preparar_base

RAIZ = Path(__file__).resolve().parents[1]
PROPORCOES = {'treino': .70, 'validacao': .15, 'teste': .15}


def validar_particoes(partes, tolerancia_tamanho=.02, tolerancia_classe=.03):
    """Falha antes de salvar/treinar se houver vazamento ou base inadequada."""
    if set(partes) != set(PROPORCOES):
        raise ValueError('São necessários treino, validacao e teste.')
    for a, b in combinations(partes, 2):
        for campo in ['id_evento', 'id_evento_original']:
            if campo in partes[a] and campo in partes[b]:
                if set(partes[a][campo]) & set(partes[b][campo]):
                    raise ValueError(f'Vazamento de evento ({campo}) entre {a} e {b}.')
    tudo = pd.concat(partes.values(), ignore_index=True)
    for campo in ['id', 'texto_original', 'texto_tfidf', 'chave_modelo']:
        if campo not in tudo or tudo[campo].isna().any() or tudo[campo].astype(str).str.strip().eq('').any():
            raise ValueError(f'Campo obrigatório vazio/ausente: {campo}.')
        if tudo[campo].duplicated().any():
            raise ValueError(f'Duplicatas em {campo}.')
    if not tudo.rotulo.isin([0, 1]).all():
        raise ValueError('Rótulos inválidos.')
    if tudo.texto_tfidf.str.contains(r'#|https?://|www\.|@', case=False).any():
        raise ValueError('Marcadores ou URLs presentes no texto do modelo.')
    for nome, df in partes.items():
        if set(df.rotulo) != {0, 1}:
            raise ValueError(f'{nome} não contém ambas as classes.')
        if abs(len(df) / len(tudo) - PROPORCOES[nome]) > tolerancia_tamanho:
            raise ValueError(f'Proporção inadequada em {nome}.')
        if abs(df.rotulo.mean() - tudo.rotulo.mean()) > tolerancia_classe:
            raise ValueError(f'Desbalanceamento de classes em {nome}.')
        for origem in tudo.dataset_origem.unique():
            sub = df[df.dataset_origem.eq(origem)]
            geral = tudo[tudo.dataset_origem.eq(origem)]
            if set(sub.rotulo) != {0, 1} or abs(sub.rotulo.mean() - geral.rotulo.mean()) > tolerancia_classe:
                raise ValueError(f'Desbalanceamento por origem em {nome}/{origem}.')
    fakebr = tudo[tudo.dataset_origem.eq('Fake.Br')].copy()
    fakebr['par'] = fakebr.id.str.rsplit('_', n=1).str[-1]
    if not fakebr.groupby('par').id_evento.nunique().eq(1).all():
        raise ValueError('Par Fake.Br com eventos diferentes.')


def dividir_base(df, random_state=42, tentativas=128):
    """Escolhe divisão apenas pela composição, nunca pelo desempenho do modelo.

    GroupShuffleSplit amostra grupos, não linhas: avaliamos desvios por linha e
    distribuição conjunta de origem/classe em 128 candidatos determinísticos.
    Os grupos nunca são quebrados. Se nenhum candidato atende, a execução falha.
    """
    if df.id_evento.nunique() < 7:
        raise ValueError('Poucos eventos para dividir em três conjuntos.')
    estratos = df.dataset_origem.astype(str) + ':' + df.rotulo.astype(str)
    matriz = pd.get_dummies(estratos).to_numpy(dtype=float)
    global_ = matriz.mean(axis=0)
    melhor, menor, escolhido = None, float('inf'), None
    grupos = df.id_evento.to_numpy()
    for tentativa in range(tentativas):
        seed = random_state + tentativa
        treino, resto = next(GroupShuffleSplit(n_splits=1, train_size=.7, random_state=seed).split(df, groups=grupos))
        iv, it = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed).split(resto, groups=grupos[resto]))
        indices = {'treino': treino, 'validacao': resto[iv], 'teste': resto[it]}
        custo = sum(abs(len(idx) / len(df) - PROPORCOES[n]) + np.abs(matriz[idx].mean(axis=0) - global_).sum()
                    for n, idx in indices.items())
        if custo < menor:
            candidato = {n: df.iloc[idx].copy() for n, idx in indices.items()}
            # Checagem rápida de tamanho/classes; a auditoria completa ocorre ao final.
            adequado = all(abs(len(p)/len(df)-PROPORCOES[n]) <= .02 and set(p.rotulo) == {0, 1}
                           and abs(p.rotulo.mean()-df.rotulo.mean()) <= .03
                           and all(set(p[p.dataset_origem.eq(o)].rotulo) == {0, 1}
                                   and abs(p[p.dataset_origem.eq(o)].rotulo.mean()-df[df.dataset_origem.eq(o)].rotulo.mean()) <= .03
                                   for o in df.dataset_origem.unique())
                           for n, p in candidato.items())
            if adequado:
                melhor, menor, escolhido = candidato, float(custo), seed
    if melhor is None:
        raise ValueError('Nenhuma divisão adequada; revise o tamanho dos grupos sem separá-los.')
    validar_particoes(melhor)
    return melhor, {'random_state': random_state, 'semente_candidata': escolhido,
                    'tentativas': tentativas, 'custo_distribuicao': menor}


def sha256_arquivo(caminho):
    digest = hashlib.sha256()
    with Path(caminho).open('rb') as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b''):
            digest.update(bloco)
    return digest.hexdigest()


def main():
    entrada = RAIZ / 'dados_processados/noticias_limpas.csv'
    resultados = RAIZ / 'resultados'
    saidas = {n: entrada.parent / f'{n}.csv' for n in PROPORCOES}
    protegidos = [*saidas.values(), resultados / 'divisao.json', resultados / 'avaliacao_final.json']
    if any(p.exists() for p in protegidos):
        raise FileExistsError('Divisão/avaliação já existe. Preserve o experimento; não sobrescreva seus conjuntos.')
    base = pd.read_csv(entrada, keep_default_na=False)
    df, auditoria, removidos = preparar_base(base)
    partes, busca = dividir_base(df)
    resultados.mkdir(exist_ok=True)
    manifesto = {'preparacao': auditoria, 'busca': busca, 'sha256_entrada': sha256_arquivo(entrada),
                 'tolerancia_tamanho': .02, 'tolerancia_classe': .03, 'conjuntos': {}}
    for nome, parte in partes.items():
        parte.to_csv(saidas[nome], index=False, encoding='utf-8-sig')
        contagens = pd.crosstab(parte.dataset_origem, parte.rotulo)
        manifesto['conjuntos'][nome] = {'linhas': len(parte), 'fracao': len(parte)/len(df),
            'eventos': int(parte.id_evento.nunique()), 'sha256': sha256_arquivo(saidas[nome]),
            'distribuicao': {o: {str(r): int(v) for r, v in linha.items()} for o, linha in contagens.to_dict('index').items()}}
        print(f'\n{nome}: {len(parte)} notícias ({len(parte)/len(df):.2%})')
        print(contagens.rename(columns={0: 'falso', 1: 'verdadeiro'}))
    removidos.to_csv(resultados / 'registros_excluidos_divisao.csv', index=False, encoding='utf-8-sig')
    (resultados / 'divisao.json').write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nInterseções de id_evento: treino/validação=0, treino/teste=0, validação/teste=0.')
    print('Preparação:', auditoria)
    print('Busca:', busca)


if __name__ == '__main__':
    main()
