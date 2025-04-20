# Análise de Preferências em Catálogo de Streaming com APRIORI

Este projeto implementa um sistema de análise de preferências para plataformas de streaming utilizando o algoritmo APRIORI para descobrir padrões de associação entre os conteúdos assistidos pelos usuários.

## Funcionalidades

- Geração de dataset simulado de visualizações de streaming
- Análise de regras de associação utilizando o algoritmo APRIORI
- Formatação das regras em linguagem natural (ex: "Quem assistiu X também pode gostar de Y")
- Visualização gráfica das melhores regras encontradas
- Salvamento dos resultados em arquivo de texto

## Requisitos

Para executar o projeto, você precisará instalar as seguintes bibliotecas Python:

```
pandas
numpy
mlxtend
matplotlib
seaborn
```

Você pode instalar todas as dependências usando:

```
pip install pandas numpy mlxtend matplotlib seaborn
```

## Como usar

1. Clone este repositório
2. Instale as dependências necessárias
3. Execute o script principal:

```
python analise_preferencias_streaming.py
```

## Resultados

Ao executar o programa, você obterá:

1. Um dataset simulado de visualizações de streaming
2. Análise estatística básica dos dados
3. Regras de associação encontradas (ex: "Usuários que assistem 'Breaking Bad' e 'Narcos' também assistem 'Peaky Blinders'")
4. Gráfico visual das principais regras (salvo como 'top_regras_associacao.png')
5. Arquivo de texto com todas as recomendações geradas (salvo como 'recomendacoes_streaming.txt')

## Personalização

Você pode ajustar os parâmetros do algoritmo no código:

- `min_support`: Suporte mínimo (proporção de usuários que seguem o padrão)
- `min_confidence`: Confiança mínima (probabilidade condicional)
- `min_lift`: Lift mínimo (medida de independência entre itens)
- `num_usuarios`: Número de usuários no dataset simulado
- `num_titulos`: Número de títulos no catálogo simulado

## Sobre o algoritmo APRIORI

O algoritmo APRIORI é uma técnica de mineração de dados utilizada para descobrir padrões frequentes em grandes conjuntos de dados. No contexto deste projeto, ele é usado para identificar quais conteúdos são frequentemente assistidos juntos pelos usuários, permitindo gerar recomendações personalizadas como "Se você gostou de X, também pode gostar de Y".
