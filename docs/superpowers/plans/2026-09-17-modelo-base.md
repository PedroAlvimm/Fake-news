# Plano: divisão e modelo-base

Objetivo: concluir as sete etapas do pedido anexado, preservando a base existente.
Arquitetura: preparação conservadora e agrupamento antes da divisão; TF-IDF e
regressão logística ajustados exclusivamente no treino; validação antes da
avaliação final; análise posterior das previsões salvas.
Tecnologias: Python da .venv, pandas, scikit-learn, pytest, matplotlib, joblib.

## Restrições
- Semente 42; alvos 70/15/15; tolerância de 2 pontos percentuais no tamanho.
- Classe 0 = falsa; classe 1 = verdadeira; métricas explícitas para ambas.
- Preservar noticias_limpas.csv, texto_original e texto_bert; sem nova lematização.
- Nenhum metadado como atributo. texto_tfidf será refeito somente a partir do corpo.
- Pares Fake.Br e correspondências exatas de corpo/título pertencem ao mesmo grupo.
- Não declarar ausência de relações semânticas desconhecidas no FakeRecogna.
- Não reavaliar teste para ajustar modelo; bloquear execução acidental repetida.

## Entregas e verificações
- [ ] `tests/test_divisao.py`: invariantes sintéticas e integração com os três CSVs;
  nenhuma interseção, duas classes, textos e IDs válidos, duplicatas ausentes,
  cobertura, proporções e pares preservados. Executar antes da implementação.
- [ ] `src/preparar_divisao.py`: retirar hashtags, URLs, emails, fontes explícitas e
  prefixos editoriais; agrupar transitivamente eventos/corpos/títulos; registrar
  remoções e conflitos sem alterar o CSV de entrada.
- [ ] `src/dividir_dados.py`: testar 128 divisões reproduzíveis por grupo, selecionar
  pelo desvio de tamanho e distribuição conjunta origem/classe; validar antes de
  gravar os CSVs, manifesto JSON e relatório de exclusões.
- [ ] `tests/test_modelo.py`: verificar classe de interesse 0, AUC de subconjunto
  de classe única, vocabulário aprendido só no treino e nomes dos tipos de erro.
- [ ] `src/treinar_modelo_base.py`: unigramas/bigramas, min_df=3, 50000 atributos,
  sublinear_tf; regressão logística C=1, semente 42, max_iter=1000. Usar pesos só
  se razão maior/menor >= 1.5 no treino. Limiar fixo 0.5. Salvar avaliação da
  validação e congelar configuração antes de prever o teste uma única vez.
- [ ] `src/analisar_erros.py`: juntar previsões ao teste por id, validar relação 1:1,
  salvar erros ordenados por confiança e resumos por origem e categoria.
- [ ] Documentar instalação, comandos, resultados reais e limitações no README.
- [ ] Executar divisão, pytest, treinamento, análise e verificar artefatos salvos.

Não existe repositório .git na raiz: nenhuma alteração será commitada.
