import hashlib
import html
import re
import unicodedata
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from datasets import load_dataset


# =========================================================
# CAMINHOS DO PROJETO
# =========================================================

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

PASTA_FAKEBR = (
    RAIZ_PROJETO
    / "dados_brutos"
    / "Fake.br-Corpus"
    / "full_texts"
)

PASTA_SAIDA = RAIZ_PROJETO / "dados_processados"

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)


# =========================================================
# FUNÇÕES DE LIMPEZA
# =========================================================

def limpar_texto_basico(texto):
    """
    Limpeza moderada para BERTimbau e Transformers.

    Preserva:
    - acentos;
    - números;
    - pontuação;
    - palavras de negação;
    - letras maiúsculas e minúsculas.
    """

    if texto is None or pd.isna(texto):
        return ""

    texto = str(texto)

    # Converte códigos HTML, como &amp;
    texto = html.unescape(texto)

    # Remove tags HTML
    texto = BeautifulSoup(
        texto,
        "html.parser"
    ).get_text(separator=" ")

    # Padroniza os caracteres Unicode
    texto = unicodedata.normalize("NFC", texto)

    # Remove caracteres invisíveis
    texto = re.sub(
        r"[\u200b-\u200d\uFEFF]",
        "",
        texto
    )

    # Substitui links encontrados dentro do texto
    texto = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        texto,
        flags=re.IGNORECASE
    )

    # Substitui endereços de e-mail
    texto = re.sub(
        r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        " EMAIL ",
        texto
    )

    # Remove o marcador explícito que pode entregar o rótulo
    texto = re.sub(
        r"#\s*boato\b",
        " ",
        texto,
        flags=re.IGNORECASE
    )

    # Remove espaços e quebras de linha excessivas
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def preparar_para_tfidf(texto):
    """
    Prepara o texto para modelos clássicos:
    - TF-IDF;
    - regressão logística;
    - SVM;
    - Naive Bayes.
    """

    texto = limpar_texto_basico(texto)

    # Padroniza em letras minúsculas
    texto = texto.lower()

    # Mantém acentos, números e pontuação importante
    texto = re.sub(
        r"[^a-záàâãéèêíïóôõöúüç0-9!?., ]",
        " ",
        texto
    )

    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def criar_chave_duplicata(texto):
    """
    Cria uma chave para detectar notícias idênticas.
    """

    texto = limpar_texto_basico(texto).lower()

    # Remove pontuação apenas para comparar duplicatas
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return hashlib.sha256(
        texto.encode("utf-8")
    ).hexdigest()


# =========================================================
# CARREGAMENTO DO FAKE.BR
# =========================================================

def carregar_pasta_fakebr(pasta, rotulo, classe):
    registros = []

    if not pasta.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {pasta}"
        )

    arquivos = sorted(pasta.glob("*.txt"))

    print(
        f"Encontrados {len(arquivos)} arquivos "
        f"na pasta {pasta.name}"
    )

    for arquivo in arquivos:
        texto = arquivo.read_text(
            encoding="utf-8",
            errors="replace"
        )

        registros.append({
            "id": f"fakebr_{classe}_{arquivo.stem}",
            "id_evento": f"fakebr_evento_{arquivo.stem}",
            "titulo": "",
            "subtitulo": "",
            "texto_original": texto,
            "categoria": "",
            "data": "",
            "autor": "",
            "url": "",
            "rotulo": rotulo,
            "classe": classe,
            "dataset_origem": "Fake.Br"
        })

    return registros


def carregar_fakebr():
    print("\nCarregando Fake.Br...")

    registros_falsos = carregar_pasta_fakebr(
        PASTA_FAKEBR / "fake",
        rotulo=0,
        classe="falso"
    )

    registros_verdadeiros = carregar_pasta_fakebr(
        PASTA_FAKEBR / "true",
        rotulo=1,
        classe="verdadeiro"
    )

    dataframe = pd.DataFrame(
        registros_falsos + registros_verdadeiros
    )

    print(
        f"Fake.Br carregado: {len(dataframe)} registros"
    )

    return dataframe


# =========================================================
# CARREGAMENTO DO FAKERECOGNA
# =========================================================

def carregar_fakerecogna():
    print("\nBaixando ou carregando FakeRecogna...")

    dataset = load_dataset(
        "recogna-nlp/FakeRecogna",
        split="train"
    )

    dataframe = dataset.to_pandas()

    # Padroniza os nomes das colunas
    dataframe = dataframe.rename(columns={
        "Titulo": "titulo",
        "Subtitulo": "subtitulo",
        "Noticia": "texto_original",
        "Categoria": "categoria",
        "Data": "data",
        "Autor": "autor",
        "URL": "url",
        "Classe": "rotulo"
    })

    colunas_textuais = [
        "titulo",
        "subtitulo",
        "texto_original",
        "categoria",
        "data",
        "autor",
        "url"
    ]

    # Preenche campos ausentes
    for coluna in colunas_textuais:
        if coluna not in dataframe.columns:
            dataframe[coluna] = ""

        dataframe[coluna] = (
            dataframe[coluna]
            .fillna("")
            .astype(str)
        )

    # Converte os rótulos para números
    dataframe["rotulo"] = pd.to_numeric(
        dataframe["rotulo"],
        errors="coerce"
    )

    # Mantém somente rótulos válidos
    dataframe = dataframe[
        dataframe["rotulo"].isin([0, 1])
    ].copy()

    dataframe["rotulo"] = (
        dataframe["rotulo"]
        .astype(int)
    )

    dataframe["classe"] = dataframe["rotulo"].map({
        0: "falso",
        1: "verdadeiro"
    })

    dataframe = dataframe.reset_index(drop=True)

    dataframe["id"] = [
        f"fakerecogna_{indice}"
        for indice in dataframe.index
    ]

    dataframe["id_evento"] = dataframe["id"]
    dataframe["dataset_origem"] = "FakeRecogna"

    colunas_finais = [
        "id",
        "id_evento",
        "titulo",
        "subtitulo",
        "texto_original",
        "categoria",
        "data",
        "autor",
        "url",
        "rotulo",
        "classe",
        "dataset_origem"
    ]

    dataframe = dataframe[colunas_finais]

    print(
        f"FakeRecogna carregado: {len(dataframe)} registros"
    )

    return dataframe


# =========================================================
# PREPARAÇÃO DOS TEXTOS
# =========================================================

def preparar_dataframe(dataframe):
    dataframe = dataframe.copy()

    # Junta título, subtítulo e notícia
    dataframe["texto_completo"] = (
        "[TITULO] "
        + dataframe["titulo"].fillna("").astype(str)
        + " [SUBTITULO] "
        + dataframe["subtitulo"].fillna("").astype(str)
        + " [TEXTO] "
        + dataframe["texto_original"].fillna("").astype(str)
    )

    # Versão para BERTimbau
    dataframe["texto_bert"] = (
        dataframe["texto_completo"]
        .apply(limpar_texto_basico)
    )

    # Versão para TF-IDF
    dataframe["texto_tfidf"] = (
        dataframe["texto_completo"]
        .apply(preparar_para_tfidf)
    )

    # Medidas para controle de qualidade
    dataframe["quantidade_caracteres"] = (
        dataframe["texto_bert"].str.len()
    )

    dataframe["quantidade_palavras"] = (
        dataframe["texto_bert"]
        .str.split()
        .str.len()
    )

    # Chave usada para localizar duplicatas
    dataframe["chave_duplicata"] = (
        dataframe["texto_bert"]
        .apply(criar_chave_duplicata)
    )

    return dataframe


# =========================================================
# VALIDAÇÃO E REMOÇÃO DE DUPLICADOS
# =========================================================

def validar_e_remover_duplicados(dataframe):
    dataframe = dataframe.copy()

    quantidade_inicial = len(dataframe)

    # Retira textos vazios ou muito pequenos
    dataframe = dataframe[
        dataframe["quantidade_caracteres"] >= 50
    ].copy()

    # Procura textos iguais com rótulos diferentes
    quantidade_rotulos = (
        dataframe
        .groupby("chave_duplicata")["rotulo"]
        .nunique()
    )

    chaves_conflitantes = quantidade_rotulos[
        quantidade_rotulos > 1
    ].index

    conflitos = dataframe[
        dataframe["chave_duplicata"].isin(
            chaves_conflitantes
        )
    ].copy()

    if not conflitos.empty:
        arquivo_conflitos = (
            PASTA_SAIDA
            / "registros_conflitantes.csv"
        )

        conflitos.to_csv(
            arquivo_conflitos,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"Conflitos encontrados: {len(conflitos)}"
        )

        # Retira exemplos contraditórios da base principal
        dataframe = dataframe[
            ~dataframe["chave_duplicata"].isin(
                chaves_conflitantes
            )
        ].copy()

    # Mantém somente uma cópia de textos iguais
    dataframe = dataframe.drop_duplicates(
        subset=["chave_duplicata"],
        keep="first"
    )

    quantidade_final = len(dataframe)

    print(
        "Registros retirados durante a validação:",
        quantidade_inicial - quantidade_final
    )

    return dataframe.reset_index(drop=True)


# =========================================================
# RELATÓRIO DA BASE
# =========================================================

def mostrar_relatorio(dataframe):
    print("\nQuantidade por base e classe:")

    print(
        pd.crosstab(
            dataframe["dataset_origem"],
            dataframe["classe"]
        )
    )

    print("\nEstatísticas de tamanho dos textos:")

    print(
        dataframe["quantidade_palavras"]
        .describe()
        .round(2)
    )

    print("\nValores ausentes em colunas importantes:")

    print(
        dataframe[
            [
                "texto_bert",
                "texto_tfidf",
                "rotulo",
                "classe",
                "dataset_origem"
            ]
        ]
        .isna()
        .sum()
    )


# =========================================================
# EXECUÇÃO PRINCIPAL
# =========================================================

def main():
    fakebr = carregar_fakebr()
    fakerecogna = carregar_fakerecogna()

    print("\nJuntando as bases...")

    dataframe = pd.concat(
        [fakebr, fakerecogna],
        ignore_index=True
    )

    print(
        f"Total antes da limpeza: {len(dataframe)}"
    )

    dataframe = preparar_dataframe(dataframe)

    dataframe = validar_e_remover_duplicados(
        dataframe
    )

    arquivo_saida = (
        PASTA_SAIDA
        / "noticias_limpas.csv"
    )

    dataframe.to_csv(
        arquivo_saida,
        index=False,
        encoding="utf-8-sig"
    )

    mostrar_relatorio(dataframe)

    print("\nLimpeza finalizada com sucesso!")
    print(f"Arquivo criado em: {arquivo_saida}")
    print(f"Total final: {len(dataframe)} registros")


if __name__ == "__main__":
    main()