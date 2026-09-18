# Artefatos e Reprodutibilidade

Este documento descreve como reproduzir os artefatos gerados por este projeto (modelos, vetorizadores, métricas e relatórios) e fornece ferramentas úteis para registrar metadados e checksums.

Objetivos
- Garantir que terceiros (ou você no futuro) possam recriar um modelo e seus resultados.
- Rastrear versões de artefatos e validar integridade via checksums.

Requisitos básicos
- Sistema: macOS / Linux / Windows (com adaptadores para comandos). 
- Python 3.10+ e `pip`.
- Espaço em disco suficiente para armazenar os dados e artefatos.

1) Ambiente reprodutível

Crie e ative um ambiente virtual, instale dependências e trave a lista de pacotes:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip freeze > requirements-frozen.txt
```

Guarde `requirements-frozen.txt` junto com o commit dos resultados quando publicar um relatório ou modelo.

2) Dados e divisão

- Os dados brutos ficam em `dados_brutos/` (não versionados).
- Execute `python3 src/dividir_dados.py` para gerar as divisões em `dados_processados/`.
- O arquivo `resultados/divisao.json` contém metadados e checksums da divisão gerada; conserve-o para rastreabilidade.

3) Treino e artefatos

- Treine o modelo base com:
```bash
python3 src/treinar_modelo_base.py
```
- Artefatos salvos em `modelos/` (por exemplo, `vetorizador_tfidf.joblib`, `modelo_tfidf_logistica.joblib`).
- Para busca de hiperparâmetros leve:
```bash
python3 src/busca_hiperparametros.py
```

4) Registrar checksums dos artefatos (script automatizado)

Use o script `scripts/record_checksums.py` para gerar `artifacts_checksums.json` com sha256 e metadados para todos os arquivos dentro de `modelos/` e `resultados/`.

```bash
python3 scripts/record_checksums.py --output resultados/artifacts_checksums.json
```

Exemplo de saída (JSON): chave = caminho relativo, valor = {sha256, size_bytes, mtime}.

5) Boas práticas de versionamento de artefatos
- Não mantenha artefatos grandes no repositório Git; use `.gitignore` (já configurado) ou Git LFS.
- Quando publicar um resultado, inclua:
  - `requirements-frozen.txt`
  - `resultados/divisao.json`
  - `resultados/artifacts_checksums.json`
  - `README.md` ou `REPRODUCIBILITY.md` com os passos usados

6) Recriar um experimento do zero

Passos resumidos:
```bash
git clone <repo>
cd <repo>
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
# colocar os dados brutos em dados_brutos/
python3 src/dividir_dados.py
python3 src/treinar_modelo_base.py
python3 scripts/record_checksums.py --output resultados/artifacts_checksums.json
```

7) Reprodutibilidade determinística (notas)
- Para máxima determinismo com scikit-learn, garanta que os parâmetros `random_state` sejam fixos nos scripts (ex.: nos classificadores e nas divisões). Alguns pontos a considerar:
  - Defina manualmente `random_state` no código (`src/dividir_dados.py`, `src/treinar_modelo_base.py`).
  - Controle `PYTHONHASHSEED` se necessário: `export PYTHONHASHSEED=0`.
  - Versões de pacotes (registradas em `requirements-frozen.txt`) influenciam o comportamento; sempre arquive essa lista.

8) Metadados opcionais
- Para maior rastreabilidade, pode-se adicionar `modelos/manifest.json` com: versão do modelo, parâmetros usados, data/hora, autor, commit git (hash) e checksum do arquivo do modelo.

9) Contato
Se quiser, eu crio o `modelos/manifest.json` automaticamente após o treino, ou adapto `src/treinar_modelo_base.py` para gravar metadados/seed no momento do salvamento.

---
Arquivo relacionado: `scripts/record_checksums.py` (gera checksums para `modelos/` e `resultados/`).
