# Fake News Dataset — Pipeline

Projeto: pipeline para treinar, avaliar e inferir modelos de classificação de notícias (falso/verdadeiro) usando TF‑IDF + Regressão Logística. Inclui ferramentas para pré-processamento, divisão de dados preservando grupos, busca leve de hiperparâmetros, análise de erros e interface de inferência (CLI + Streamlit).

Este repositório foi organizado para ser reprodutível e leve — dados brutos, modelos e resultados grandes são mantidos fora do controle de versão por padrão via `.gitignore`.

## Sumário
- Descrição rápida
- Requisitos e instalação
- Execução (preparação, treino, busca, análise)
- Inferência (CLI e Streamlit)
- Estrutura do projeto
- Como publicar no GitHub
- Contato e contribuições

## Requisitos
- Python 3.10+ (recomendado)
- Espaço em disco suficiente para os dados (`dados_brutos/` pode ser grande)

Recomendo o uso de um ambiente virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Execução — comandos principais

1) Testes (sanidade)
```bash
python3 -m pytest -q
```

2) Preparar e dividir dados
- `src/preparar_divisao.py` contém funções de limpeza/normalização do texto.
- Execute:
```bash
python3 src/dividir_dados.py
```
Isto gera `dados_processados/treino.csv`, `dados_processados/validacao.csv`, `dados_processados/teste.csv` e `resultados/divisao.json` (com checksums para rastreabilidade).

3) Treinar modelo base
```bash
python3 src/treinar_modelo_base.py
```
Saídas: artefatos em `modelos/` (vetorizador + modelo) e métricas/relatórios em `resultados/`.

4) Busca leve de hiperparâmetros (opcional)
```bash
python3 src/busca_hiperparametros.py
```
Pode gerar `modelos/modelo_tuned.joblib` e `modelos/vetorizador_tuned.joblib`.

5) Análise de erros
```bash
python3 src/analisar_erros.py
```
Gera `resultados/analise_erros.csv` com linhas de erro ordenadas por confiança e sumarizações por origem/categoria.

## Inferência

- CLI (arquivo CSV com coluna `texto` ou texto único):
```bash
python3 src/inferencia_cli.py --input dados_processados/teste.csv --output previsoes_demo.csv
python3 src/inferencia_cli.py --text "Notícia de exemplo sobre política"
```

- App visual (Streamlit):
```bash
streamlit run src/app_streamlit.py
```

Observações sobre inferência:
- O CSV de entrada deve ter a coluna `texto`. Se não tiver, renomeie ou ajuste o arquivo.
- O CSV de saída contém as colunas originais mais `probabilidade_verdadeiro` e `rotulo_previsto`.
- Para usar um modelo não-padrão, passe `--model` e `--vet` ao CLI ou ajuste os caminhos no app Streamlit.

## Estrutura do repositório (resumida)

- `dados_brutos/` — dados originais (não versionados).  
- `dados_processados/` — saídas de preparação/divisão (não versionadas).  
- `modelos/` — vetorizadores e modelos salvos (não versionados).  
- `resultados/` — métricas, gráficos e relatórios (não versionados).  
- `src/` — scripts Python principais:  
  - `preparar_divisao.py` — limpeza e normalização de texto.  
  - `dividir_dados.py` — divisão por `id_evento` com validação e checksums.  
  - `treinar_modelo_base.py` — treina TF‑IDF + LogisticRegression e salva artefatos.  
  - `busca_hiperparametros.py` — busca leve sobre parâmetros do pipeline.  
  - `analisar_erros.py` — gera `analise_erros.csv` e soma por origem/categoria.  
  - `inferencia_cli.py` — CLI para inferência em lote/única.  
  - `app_streamlit.py` — interface web para inferência interativa.

## Publicação no GitHub — passo a passo recomendado

1. Verifique o `.gitignore` (ele já inclui `dados_brutos/`, `dados_processados/`, `modelos/`, `resultados/`).

2. Se houver arquivos grandes já rastreados, remova-os do índice antes do commit:
```bash
git rm -r --cached dados_brutos dados_processados modelos resultados || true
git rm --cached .env || true
```

3. Inicialize o repositório e faça o commit do código:
```bash
git init
git config user.name "Seu Nome"
git config user.email "seu@exemplo.com"
git add .
git commit -m "Initial commit: código (sem dados/modelos)"
```

4. Criar repositório remoto com GitHub CLI e dar push (recomendado):
```bash
gh repo create meu-repo --public --source=. --remote=origin --push
```
Se preferir criar pelo site do GitHub, crie o repositório e depois:
```bash
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPO.git
git push -u origin main
```

5. Depois de publicar, cole o link do repositório aqui para que eu faça uma revisão rápida (opcional).

## Boas práticas antes de publicar

- Remova segredos/credenciais (`.env`, tokens).  
- Inclua `requirements.txt` atualizado.  
- Opcional: adicione `LICENSE` e `CONTRIBUTING.md`.  
- Verifique se `README.md` documenta os passos mínimos para executar o projeto.

## Licença
Adicione um arquivo `LICENSE` se pretende compartilhar publicamente. Recomendo `MIT` para máxima simplicidade, ou `Apache-2.0` para proteção de patentes.

## Contato / Próximos passos

Se quiser, eu posso:
- (A) executar aqui o `git init` + commit local (não faço push ao remoto sem sua permissão), ou
- (B) tentar criar o repositório remoto com `gh` (requere `gh` autenticado no terminal), ou
- (C) gerar um arquivo ZIP com apenas o código pronto para upload.

Envie a opção desejada ou o link do GitHub quando pronto e eu faço a revisão.
