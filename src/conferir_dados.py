from pathlib import Path

import pandas as pd


RAIZ_PROJETO = Path(__file__).resolve().parent.parent

ARQUIVO = (
    RAIZ_PROJETO
    / "dados_processados"
    / "noticias_limpas.csv"
)


def main():
    if not ARQUIVO.exists():
        print("ERRO: noticias_limpas.csv não foi encontrado.")
        print(f"Caminho procurado: {ARQUIVO}")
        return

    df = pd.read_csv(ARQUIVO)

    print("\nBASE CARREGADA COM SUCESSO")
    print("=" * 50)

    print(f"\nQuantidade de linhas: {len(df)}")
    print(f"Quantidade de colunas: {len(df.columns)}")

    print("\nColunas encontradas:")
    for coluna in df.columns:
        print(f"- {coluna}")

    print("\nQuantidade por classe:")
    print(df["classe"].value_counts(dropna=False))

    print("\nQuantidade por base:")
    print(df["dataset_origem"].value_counts(dropna=False))

    print("\nCruzamento entre base e classe:")
    print(
        pd.crosstab(
            df["dataset_origem"],
            df["classe"]
        )
    )

    print("\nValores ausentes:")
    print(
        df[
            [
                "texto_bert",
                "texto_tfidf",
                "rotulo",
                "classe"
            ]
        ]
        .isna()
        .sum()
    )

    print("\nQuantidade de rótulos inválidos:")
    rotulos_invalidos = ~df["rotulo"].isin([0, 1])
    print(rotulos_invalidos.sum())

    print("\nDuplicatas exatas:")
    print(
        df.duplicated(
            subset=["chave_duplicata"]
        ).sum()
    )

    print("\nResumo do tamanho dos textos:")
    print(
        df["quantidade_palavras"]
        .describe()
        .round(2)
    )

    print("\nExemplo de notícia:")
    exemplo = df.iloc[0]

    print(f"Base: {exemplo['dataset_origem']}")
    print(f"Classe: {exemplo['classe']}")
    print(f"Rótulo: {exemplo['rotulo']}")

    print("\nTexto para BERT:")
    print(str(exemplo["texto_bert"])[:500])

    print("\n" + "=" * 50)

    passou = True

    if df.empty:
        print("FALHA: a base está vazia.")
        passou = False

    if df["texto_bert"].isna().any():
        print("FALHA: existem textos BERT ausentes.")
        passou = False

    if df["texto_tfidf"].isna().any():
        print("FALHA: existem textos TF-IDF ausentes.")
        passou = False

    if rotulos_invalidos.any():
        print("FALHA: existem rótulos diferentes de 0 e 1.")
        passou = False

    if df.duplicated(subset=["chave_duplicata"]).any():
        print("FALHA: ainda existem duplicatas exatas.")
        passou = False

    if passou:
        print("RESULTADO: a base passou nas verificações básicas.")
    else:
        print("RESULTADO: a base precisa ser corrigida.")


if __name__ == "__main__":
    main()