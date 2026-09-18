"""Higiene adicional para o baseline, sem modificar a base limpa de entrada."""
import hashlib
import html
import re
import unicodedata

import pandas as pd
from bs4 import BeautifulSoup


def texto_seguro(texto):
    """Usa somente conteúdo textual; mantém acentos, números e negações.

    As regras são fixas e independentes do rótulo. Não se removem palavras como
    'falso' em qualquer contexto: elas também podem pertencer à própria notícia.
    Esta higiene não substitui uma auditoria semântica de fontes e checagens.
    """
    texto = unicodedata.normalize('NFC', html.unescape(str(texto)))
    if re.search(r'</?[a-zA-Z][^>]*>', texto):
        texto = BeautifulSoup(texto, 'html.parser').get_text(' ')
    texto = re.sub(r'[\u200b-\u200d\ufeff]', '', texto)
    texto = re.sub(r'https?://\S+|www\.\S+', ' ', texto, flags=re.I)
    texto = re.sub(r'\b[\w.%+-]+@[\w.-]+\.[a-z]{2,}\b', ' ', texto, flags=re.I)
    texto = re.sub(r'#\s*boatos?\b|#[\w]+|@[\w]+', ' ', texto, flags=re.I)
    # Domínios sem protocolo e nomes explícitos de sites de checagem.
    texto = re.sub(r'\b(?:[\w-]+\.)+(?:com|org|net|br|gov|edu)(?:\.[a-z]{2})?(?:/\S*)?\b', ' ', texto, flags=re.I)
    texto = re.sub(r'\be[ -]farsas\b|\bboatos\s+org\b', ' ', texto, flags=re.I)
    texto = re.sub(r'(^|[.!?]\s*)(?:boatos?\s*[–—:-]|(?:veredito|checagem)\s*:\s*(?:falso|verdadeiro)\b)', r'\1 ', texto, flags=re.I)
    texto = re.sub(r'\b(?:URL|EMAIL)\b', ' ', texto)
    # Sem remoção de stopwords, stemming, transliteração ou nova lematização.
    # Remover hashtags e menções residuais que escaparam das regras anteriores.
    texto = re.sub(r'#[^\s]+', ' ', texto)
    texto = re.sub(r'@[^\s]+', ' ', texto)
    # Substituir qualquer caractere isolado de hashtag/arroba que tenha sobrado.
    texto = re.sub(r"[#@]", ' ', texto)
    return re.sub(r'\s+', ' ', texto).strip().lower()


def chave_texto(texto):
    normalizado = re.sub(r'[^\w\s]', ' ', texto, flags=re.UNICODE)
    normalizado = re.sub(r'\s+', ' ', normalizado).strip()
    return hashlib.sha256(normalizado.encode('utf-8')).hexdigest() if normalizado else ''


def preparar_base(base):
    """Retorna dados elegíveis, auditoria e exclusões identificáveis por id.

    Componentes conexos preservam relações transitivas ANTES da deduplicação.
    Um par Fake.Br não pode ser separado mesmo se um de seus textos unir dois
    eventos. id_evento_original permite rastrear a união dos grupos.
    """
    obrigatorias = {'id', 'id_evento', 'dataset_origem', 'rotulo', 'texto_original', 'texto_bert'}
    if faltam := obrigatorias - set(base):
        raise ValueError(f'Colunas ausentes: {sorted(faltam)}')
    df = base.copy().reset_index(drop=True)
    if df.empty or not df.rotulo.isin([0, 1]).all():
        raise ValueError('Base vazia ou rótulos fora de 0 e 1.')
    for campo in ['id', 'id_evento', 'dataset_origem']:
        if df[campo].isna().any() or df[campo].astype(str).str.strip().eq('').any():
            raise ValueError(f'{campo} contém valor vazio.')
    if df.id.duplicated().any():
        raise ValueError('IDs duplicados.')
    df['id_evento_original'] = df.id_evento
    df['texto_tfidf'] = df.texto_original.fillna('').map(texto_seguro)
    df['chave_modelo'] = df.texto_tfidf.map(chave_texto)

    pais = list(range(len(df)))

    def raiz(i):
        while pais[i] != i:
            pais[i] = pais[pais[i]]
            i = pais[i]
        return i

    def unir_coluna(valores):
        vistos = {}
        for i, valor in enumerate(valores):
            if not valor:
                continue
            if valor in vistos:
                a, b = raiz(i), raiz(vistos[valor])
                pais[max(a, b)] = min(a, b)
            else:
                vistos[valor] = i

    unir_coluna(df.id_evento_original)
    unir_coluna(df.chave_modelo)
    # Redundância deliberada: a numeração do Fake.Br também é um vínculo.
    fakebr = df.dataset_origem.eq('Fake.Br')
    numeros = df.id.str.rsplit('_', n=1).str[-1]
    unir_coluna(numeros.where(fakebr, ''))
    titulos = df.get('titulo', pd.Series('', index=df.index)).fillna('').map(texto_seguro)
    chaves_titulo = titulos.map(lambda s: chave_texto(s) if len(s) >= 30 else '')
    unir_coluna(chaves_titulo)
    # Identificadores canônicos legíveis, estáveis para esta ordem de entrada.
    df['id_evento'] = [df.at[raiz(i), 'id_evento_original'] for i in df.index]

    vazios = df.chave_modelo.eq('') | df.texto_bert.fillna('').str.strip().eq('')
    chaves_conflitantes = df.loc[~vazios].groupby('chave_modelo').rotulo.nunique()
    conflito = df.chave_modelo.isin(chaves_conflitantes[chaves_conflitantes > 1].index)
    duplicados = df.chave_modelo.duplicated(keep='first') & ~vazios & ~conflito
    motivos = pd.Series('', index=df.index)
    motivos.loc[duplicados] = 'corpo_duplicado'
    motivos.loc[conflito] = 'rotulos_conflitantes'
    motivos.loc[vazios] = 'texto_vazio'
    removidos = df.loc[motivos.ne(''), ['id', 'id_evento_original', 'id_evento', 'dataset_origem', 'rotulo']].copy()
    removidos['motivo'] = motivos.loc[removidos.index]
    saida = df.loc[motivos.eq('')].copy().reset_index(drop=True)
    auditoria = {
        'entrada': len(base), 'elegiveis': len(saida), 'removidos': len(removidos),
        'motivos': {str(k): int(v) for k, v in removidos.motivo.value_counts().items()},
        'eventos_originais': int(base.id_evento.nunique()),
        'eventos_elegiveis': int(saida.id_evento.nunique()),
        'registros_com_evento_unificado': int(df.id_evento.ne(df.id_evento_original).sum()),
        'hash_boato_na_entrada_bert': int(base.texto_bert.fillna('').str.contains(r'#\s*boatos?\b', case=False).sum()),
        'duplicatas_corpo_original': int(base.texto_original.duplicated().sum()),
        'texto_modelo': 'somente texto_original com higiene fixa; texto_bert preservado',
        'limite': 'Eventos sem correspondência exata de corpo/título podem permanecer não identificados.',
    }
    return saida, auditoria, removidos
