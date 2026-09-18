import io
from pathlib import Path
import pandas as pd
import joblib
import streamlit as st


DEFAULT_MODEL = Path("modelos/modelo_tfidf_logistica.joblib")
DEFAULT_VET = Path("modelos/vetorizador_tfidf.joblib")


@st.cache_resource
def load_artifacts(model_path: str, vet_path: str):
    model_path = Path(model_path)
    vet_path = Path(vet_path)
    if not model_path.exists() or not vet_path.exists():
        raise FileNotFoundError(f"Modelo ou vetorizador não encontrado: {model_path}, {vet_path}")
    vet = joblib.load(vet_path)
    model = joblib.load(model_path)
    return model, vet


def predict_dataframe(model, vet, df: pd.DataFrame, text_col: str = "texto", threshold: float = 0.5):
    X = vet.transform(df[text_col].astype(str))
    probs = model.predict_proba(X)[:, list(model.classes_).index(1)]
    preds = (probs >= threshold).astype(int)
    out = df.copy()
    out["probabilidade_verdadeiro"] = probs
    out["rotulo_previsto"] = preds
    return out


def main():
    st.title("Inferência — Fake News Classifier")
    st.write("Carregue um CSV com uma coluna `texto` ou escreva um texto para prever.")

    st.sidebar.header("Configuração do modelo")
    model_path = st.sidebar.text_input("Caminho do modelo", str(DEFAULT_MODEL))
    vet_path = st.sidebar.text_input("Caminho do vetorizador", str(DEFAULT_VET))
    threshold = st.sidebar.slider("Limiar (probabilidade) para rotular como verdadeiro", 0.0, 1.0, 0.5)

    # carregar artefatos
    model = vet = None
    try:
        with st.spinner("Carregando modelo e vetorizador..."):
            model, vet = load_artifacts(model_path, vet_path)
        st.success("Artefatos carregados com sucesso")
    except Exception as e:
        st.error(f"Erro ao carregar artefatos: {e}")

    st.header("Inferência de um único texto")
    texto = st.text_area("Cole o texto aqui", height=150)
    if st.button("Prever texto"):
        if model is None:
            st.error("Modelo não carregado. Verifique o caminho no painel lateral.")
        elif not texto.strip():
            st.warning("Insira algum texto para prever.")
        else:
            df = pd.DataFrame({"texto": [texto]})
            res = predict_dataframe(model, vet, df, text_col="texto", threshold=threshold)
            st.write(res.loc[:, ["texto", "probabilidade_verdadeiro", "rotulo_previsto"]])

    st.header("Inferência em lote — upload CSV")
    uploaded = st.file_uploader("Envie um CSV com coluna 'texto'", type=["csv"]) 
    sample_btn = st.button("Carregar arquivo de teste (dados_processados/teste.csv)")
    if sample_btn:
        sample_path = Path("dados_processados/teste.csv")
        if sample_path.exists():
            uploaded = sample_path
        else:
            st.error("Arquivo de teste não encontrado: dados_processados/teste.csv")

    if uploaded is not None:
        if isinstance(uploaded, (str, Path)):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_csv(io.StringIO(uploaded.getvalue().decode("utf-8")))

        if "texto" not in df.columns:
            st.error("O CSV deve conter uma coluna chamada 'texto'.")
        else:
            if model is None:
                st.error("Modelo não carregado. Verifique o caminho no painel lateral.")
            else:
                with st.spinner("Gerando previsões..."):
                    out = predict_dataframe(model, vet, df, text_col="texto", threshold=threshold)
                st.success(f"Previsões geradas: {len(out)} linhas")
                st.dataframe(out.head(100))
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("Baixar previsões (CSV)", data=csv_bytes, file_name="previsoes.csv", mime="text/csv")

    st.write("\n---\n")
    st.write("Dicas: se o carregamento falhar, verifique se os arquivos de modelo estão em `modelos/` e foram gerados pelos scripts de treino.")


if __name__ == "__main__":
    main()
