# Palmer Penguins - Pipeline de Machine Learning

## 📋 Descrição do Projeto

Este projeto implementa uma esteira completa de Machine Learning para classificação de espécies de pinguins utilizando o dataset **Palmer Penguins**. O objetivo é criar um modelo capaz de prever a espécie do pinguim (Adelie, Chinstrap ou Gentoo) com base em suas características físicas.

## 🎯 Objetivo

Desenvolver um modelo de classificação utilizando técnicas de Machine Learning para identificar automaticamente a espécie de um pinguim com base em medidas como:
- Comprimento do bico
- Profundidade do bico
- Comprimento da nadadeira
- Massa corporal
- Ilha de origem
- Sexo

## 📊 Dataset

**Fonte**: [UCI Machine Learning Repository - Palmer Penguins](https://archive.ics.uci.edu/dataset/690/palmer+penguins-3)

O dataset contém informações sobre 344 pinguins de três espécies diferentes coletadas nas ilhas Palmer, Antártica.

### Variáveis:
- `species`: Espécie do pinguim (Adelie, Chinstrap, Gentoo)
- `island`: Ilha onde foi observado (Biscoe, Dream, Torgersen)
- `bill_length_mm`: Comprimento do bico em milímetros
- `bill_depth_mm`: Profundidade do bico em milímetros
- `flipper_length_mm`: Comprimento da nadadeira em milímetros
- `body_mass_g`: Massa corporal em gramas
- `sex`: Sexo do pinguim (Male, Female)

## 🚀 Como Reproduzir a Execução

### Pré-requisitos

1. **Python 3.8+** instalado
2. **Jupyter Notebook** ou **VS Code** com extensão Python

### Instalação das Dependências

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Passo a Passo

1. **Clone o repositório**:
   ```bash
   git clone <URL_DO_SEU_REPOSITORIO>
   cd <NOME_DA_PASTA>
   ```

2. **Abra o notebook**:
   ```bash
   jupyter notebook palmer_penguins_ml_pipeline.ipynb
   ```
   
   Ou abra diretamente no VS Code.

3. **Execute as células sequencialmente**:
   - Pressione `Shift + Enter` para executar cada célula
   - Ou use "Run All" para executar todas as células de uma vez

### Estrutura do Notebook

O notebook está organizado nas seguintes seções:

1. **Importação de Bibliotecas e Carregamento dos Dados**
2. **Estatísticas Descritivas** - Análise exploratória do dataset
3. **Transformações nas Colunas** - Codificação de variáveis categóricas
4. **Transformações nas Linhas** - Remoção de valores ausentes
5. **Divisão em Treino, Validação e Teste** - Split 60/20/20
6. **Treinamento do Modelo** - Random Forest Classifier
7. **Avaliação - Matriz de Confusão e Acurácia**
8. **Predição com o Modelo Implantado** - Exemplos práticos
9. **Conclusões** - Resumo e próximos passos

## 📈 Resultados Esperados

O modelo Random Forest treinado deve alcançar:
- **Acurácia**: > 95% no conjunto de teste
- **Matriz de Confusão**: Visualização clara das predições
- **Predições**: Exemplos de classificação de novas amostras

## 🎥 Vídeo de Apresentação

[Insira aqui o link do seu vídeo de apresentação do projeto]

O vídeo demonstra:
- Execução do notebook passo a passo
- Explicação das transformações aplicadas
- Análise dos resultados obtidos
- Demonstração das predições do modelo

## 📁 Estrutura do Repositório

```
.
├── palmer_penguins_ml_pipeline.ipynb  # Notebook principal do projeto
├── projeto_02_palmer_penguins_pipeline.py  # Script Python (se aplicável)
└── README.md  # Este arquivo
```

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Pandas** - Manipulação de dados
- **NumPy** - Operações numéricas
- **Matplotlib & Seaborn** - Visualização de dados
- **Scikit-learn** - Machine Learning
  - RandomForestClassifier
  - StandardScaler
  - train_test_split
  - Métricas de avaliação

## 📝 Etapas do Pipeline

### 1. Carregamento e Exploração
- Carregamento do dataset via Seaborn
- Análise inicial das dimensões e tipos de dados

### 2. Análise Descritiva
- Estatísticas descritivas (média, mediana, desvio padrão)
- Visualizações (histogramas, boxplots)
- Análise de valores ausentes

### 3. Pré-processamento
- **Transformação de Colunas**: 
  - Label Encoding para variáveis categóricas
  - Normalização com StandardScaler
- **Transformação de Linhas**: 
  - Remoção de valores ausentes

### 4. Divisão dos Dados
- Treino: 60%
- Validação: 20%
- Teste: 20%

### 5. Treinamento
- Algoritmo: Random Forest Classifier
- 100 árvores de decisão
- Profundidade máxima: 10

### 6. Avaliação
- Matriz de confusão
- Acurácia, Precisão, Recall, F1-Score
- Análise de importância das features

### 7. Predição
- Exemplos de predições em novas amostras
- Probabilidades por classe

## 👨‍💻 Autor

**Seu Nome**
- GitHub: [seu-usuario](https://github.com/seu-usuario)
- Email: seu-email@example.com

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais como parte do Projeto 02 da disciplina de Inteligência Artificial.

## 🙏 Agradecimentos

- Dataset: Palmer Penguins via UCI Machine Learning Repository
- Biblioteca Seaborn pela disponibilização do dataset
- Professores e colegas da disciplina

---

**Data de Entrega**: 19/11/2025  
**Instituição**: [Nome da Instituição]  
**Disciplina**: Inteligência Artificial
