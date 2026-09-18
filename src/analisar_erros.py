"""Analisa previsões salvas; nunca executa o modelo novamente."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .dividir_dados import sha256_arquivo
else:
    from dividir_dados import sha256_arquivo

RAIZ = Path(__file__).resolve().parents[1]


def montar_erros(previsoes, teste):
    """Valida vínculo 1:1 e ordena erros pela probabilidade da classe prevista."""
    if previsoes.id.duplicated().any() or teste.id.duplicated().any() or set(previsoes.id) != set(teste.id):
        raise ValueError('Previsões e teste precisam ter os mesmos IDs únicos.')
    contexto = teste[['id', 'rotulo', 'dataset_origem', 'texto_original']].copy()
    contexto['categoria'] = teste.get('categoria', pd.Series('', index=teste.index))
    df = previsoes.merge(contexto, on='id', validate='one_to_one', suffixes=('', '_teste'))
    if not df.rotulo_real.eq(df.rotulo).all():
        raise ValueError('Divergência de rótulos entre previsões e teste.')
    if not df.dataset_origem.eq(df.dataset_origem_teste).all():
        raise ValueError('Divergência de origem entre previsões e teste.')
    if not df.rotulo_previsto.isin([0, 1]).all():
        raise ValueError('Rótulos previstos inválidos.')
    p = df[['probabilidade_verdadeiro', 'probabilidade_falso']].to_numpy()
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError('Probabilidades inválidas.')
    df = df[df.rotulo_real.ne(df.rotulo_previsto)].copy()
    df['tipo_erro'] = np.where(df.rotulo_real.eq(0), 'fake classificada como verdadeira', 'verdadeira classificada como fake')
    # Positivo significa verdadeiro (1) na convenção binária do arquivo.
    df['convencao_positivo_1'] = np.where(df.rotulo_real.eq(0), 'falso positivo para verdadeiro', 'falso negativo para verdadeiro')
    df['confianca'] = np.where(df.rotulo_previsto.eq(1), df.probabilidade_verdadeiro, df.probabilidade_falso)
    df['categoria'] = df.categoria.fillna('').replace(r'^\s*$', 'não informada', regex=True)
    return df.drop(columns=['rotulo', 'dataset_origem_teste']).sort_values(['confianca', 'id'], ascending=[False, True])


def main():
    pasta = RAIZ / 'resultados'
    previsoes_arquivo = pasta / 'previsoes_teste.csv'
    teste_arquivo = RAIZ / 'dados_processados/teste.csv'
    registro = json.loads((pasta / 'avaliacao_final.json').read_text(encoding='utf-8'))
    if registro['estado'] != 'concluida':
        raise ValueError('A avaliação final não foi concluída.')
    if sha256_arquivo(previsoes_arquivo) != registro['sha256_previsoes'] or sha256_arquivo(teste_arquivo) != registro['configuracao_congelada']['sha256_conjuntos']['teste']:
        raise ValueError('Previsões ou teste foram alterados após a avaliação.')
    previsoes = pd.read_csv(previsoes_arquivo, keep_default_na=False)
    teste = pd.read_csv(teste_arquivo, keep_default_na=False)
    erros = montar_erros(previsoes, teste)
    erros.to_csv(pasta / 'analise_erros.csv', index=False, encoding='utf-8-sig')
    print(f'\n{len(erros)} erros em {len(teste)} notícias de teste.')
    for coluna in ['dataset_origem', 'categoria']:
        contexto = teste.copy()
        contexto['categoria'] = contexto.get('categoria', pd.Series('', index=contexto.index)).fillna('').replace(r'^\s*$', 'não informada', regex=True)
        total = contexto.groupby(coluna).size().rename('total_noticias')
        contagem = erros.groupby(coluna).size().rename('total_erros')
        tipos = pd.crosstab(erros[coluna], erros.tipo_erro)
        resumo = total.to_frame().join(contagem).join(tipos).fillna(0)
        resumo['taxa_erro'] = resumo.total_erros / resumo.total_noticias
        resumo.to_csv(pasta / f'erros_por_{coluna}.csv', encoding='utf-8-sig')
        print(f'\nErros por {coluna}:\n{resumo.to_string()}')
    for tipo in ['fake classificada como verdadeira', 'verdadeira classificada como fake']:
        selecao = erros[erros.tipo_erro.eq(tipo)]
        print(f'\n{tipo}: {len(selecao)} casos; exemplos com maior confiança:')
        for _, linha in selecao.head(5).iterrows():
            print(f'{linha.id} | {linha.dataset_origem} | confiança={linha.confianca:.3f}')
            print(str(linha.texto_original).replace('\n', ' ')[:350])
    print('\nRelatórios salvos. Confiança do modelo não é evidência de veracidade.')


if __name__ == '__main__':
    main()
