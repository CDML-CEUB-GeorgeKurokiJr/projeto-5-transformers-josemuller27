# Análise e Agrupamento de Textos Científicos com PLN

## Visão geral

Este trabalho aplica técnicas de Processamento de Linguagem Natural sobre uma base de textos de projetos científicos escritos em português. A proposta é dupla: organizar automaticamente os projetos em grupos temáticos, sem qualquer rotulagem manual prévia, e treinar um modelo próprio de Reconhecimento de Entidades Nomeadas (NER) capaz de identificar organizações, locais e pessoas citados nos textos.

Todo o fluxo foi desenvolvido no notebook `Trabalho_José_Roberto.ipynb` e segue a seguinte sequência: limpeza e filtragem da base, anotação automática de entidades, treinamento de um NER do zero, geração de representações vetoriais com BERT, agrupamento com K-Means e visualização final dos grupos com t-SNE.

## Base de dados

A base utilizada é o arquivo `dadosTextosCientificos.tsv`, hospedado no Kaggle (dataset dl-2024). Cada registro corresponde a um projeto e traz duas colunas principais: `Título_Público` e `Descricao_pública`. O arquivo é lido com codificação latin-1 e separador de tabulação.

O caminho do arquivo está configurado para o ambiente do Kaggle (`/kaggle/input/datasets/georgekurokijr/dl-2024/dadosTextosCientificos.tsv`). Para rodar em outro ambiente, basta ajustar a variável `caminho_arquivo` na segunda célula.

## Metodologia

### 1. Pré-processamento e limpeza

O tratamento inicial padroniza as colunas de título e descrição, removendo espaços extras e preenchendo valores nulos. Em seguida, título e descrição são concatenados em um único campo de texto, com o título repetido duas vezes. Essa repetição é proposital: como o título costuma resumir o tema central do projeto, duplicá-lo dá mais peso ao assunto na hora de gerar os vetores e agrupar os textos.

Depois disso são removidas as duplicatas exatas e aplicado um filtro de ruídos. São descartados registros muito curtos (quatro palavras ou menos) e textos que indicam projetos confidenciais, não divulgados, cancelados ou com definição pendente. A comparação é feita sobre o texto normalizado, sem acentos e em minúsculas, para que variações de escrita não escapem do filtro. Ao final, o notebook informa quantos registros restaram após a limpeza.

### 2. Normalização e remoção de stopwords

Para a etapa de análise dos temas, cria-se uma versão simplificada de cada texto: tudo em minúsculas, apenas letras (pontuação e números são removidos) e sem stopwords. Além da lista padrão de stopwords em português do NLTK, foram adicionadas palavras muito frequentes nesse tipo de documento, como "desenvolvimento", "projeto", "objetivo", "estudo" e "empresa", que aparecem em quase todos os registros e não ajudam a diferenciar os assuntos. Palavras com duas letras ou menos também são descartadas.

### 3. Anotação automática de entidades

Como a base não possui anotações de entidades, o modelo pré-treinado `pt_core_news_lg` do spaCy é usado como anotador: ele percorre todos os textos e marca as entidades que encontra (pessoas, organizações, locais e outras categorias). Foi incluída uma correção heurística para um erro recorrente do modelo base, garantindo que a palavra "Brasil" seja sempre classificada como local (LOC). Os textos anotados formam o conjunto de treino da etapa seguinte.

### 4. Treinamento do modelo NER customizado

Com o conjunto anotado em mãos, cria-se um pipeline em branco do spaCy para o português contendo apenas o componente de NER. Todos os rótulos observados na anotação são registrados e o modelo é treinado por 10 épocas, com embaralhamento dos dados a cada época e dropout de 0.3 para reduzir o risco de decorar os exemplos. A perda (loss) é exibida ao final de cada época, o que permite acompanhar a convergência do treinamento.

### 5. Geração de embeddings com BERT

Para representar o conteúdo semântico dos textos, é utilizado o BERTimbau (`neuralmind/bert-base-portuguese-cased`), um modelo BERT treinado especificamente para o português. Cada texto é tokenizado com limite de 512 tokens e passa pelo modelo sem cálculo de gradiente. O vetor final de cada projeto é obtido pela média dos estados ocultos da última camada, resultando em um embedding de dimensão fixa por documento. O código detecta automaticamente se há GPU disponível e a utiliza para acelerar o processamento.

### 6. Agrupamento com K-Means

Os embeddings de todos os projetos são empilhados em uma matriz e agrupados com o algoritmo K-Means em 5 clusters, com `random_state=42` para garantir reprodutibilidade. Ao final, uma tabela mostra quantos projetos ficaram em cada grupo, permitindo verificar se a divisão ficou equilibrada ou se algum tema concentra a maior parte da base.

### 7. Visualização com t-SNE e nomeação dos grupos

Antes de plotar, cada cluster recebe um nome automático: aplica-se TF-IDF sobre os textos limpos de cada grupo e extrai-se a palavra mais representativa, que passa a identificar o tema daquele cluster. Caso um grupo não tenha palavras válidas, ele recebe o rótulo "GERAL".

A redução de dimensionalidade é feita com t-SNE em duas dimensões, usando `perplexity=30` e inicialização por PCA, escolha que melhora bastante a separação visual entre os grupos. O gráfico final mostra cada projeto como um ponto colorido de acordo com seu cluster, com uma caixa de texto posicionada na mediana de cada grupo exibindo o número e o tema identificado.

### 8. Teste do modelo NER treinado

Na última etapa, o modelo NER treinado do zero é aplicado sobre cinco textos da base. Para cada um, o notebook imprime um trecho do texto e a lista de entidades encontradas com seus respectivos rótulos, servindo como verificação qualitativa do que o modelo aprendeu.

## Tecnologias utilizadas

- Python 3
- pandas e numpy para manipulação dos dados
- NLTK para stopwords em português
- spaCy para anotação e treinamento do NER (`pt_core_news_lg`)
- Transformers (Hugging Face) e PyTorch para os embeddings com BERTimbau
- scikit-learn para K-Means, TF-IDF e t-SNE
- matplotlib para a visualização final

## Como executar

1. Abra o notebook em um ambiente com suporte a GPU, como Kaggle ou Google Colab. A GPU não é obrigatória, mas acelera bastante a geração dos embeddings.
2. Execute a primeira célula, que instala todas as dependências e baixa o modelo `pt_core_news_lg` do spaCy e os recursos do NLTK.
3. Ajuste a variável `caminho_arquivo` na segunda célula caso o arquivo `dadosTextosCientificos.tsv` esteja em outro local.
4. Execute as células na ordem em que aparecem. As etapas de geração de embeddings e de t-SNE são as mais demoradas, e o tempo total varia conforme o tamanho da base e o hardware disponível.

## Resultados esperados

Ao final da execução, o notebook produz:

- a contagem de registros válidos após a limpeza de ruídos;
- o histórico de perda do treinamento do NER, época a época;
- uma tabela com a distribuição de projetos por cluster;
- um gráfico de dispersão em duas dimensões com os 5 grupos temáticos identificados e nomeados automaticamente;
- exemplos de extração de entidades feitos pelo modelo NER treinado no próprio trabalho.

## Observações

O uso de `random_state=42` no K-Means e no t-SNE garante que os resultados sejam reproduzíveis entre execuções. O número de clusters foi fixado em 5, mas pode ser ajustado na variável `num_clusters` para explorar outras granularidades de agrupamento. Da mesma forma, a lista de stopwords customizadas e as regras do filtro de ruídos podem ser adaptadas caso a base de dados seja substituída por outra de natureza diferente.
