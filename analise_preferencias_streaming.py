"""
Análise de Preferências em Catálogo de Streaming com APRIORI

Este script implementa o algoritmo Apriori para analisar preferências de usuários
em plataformas de streaming, gerando regras de associação para recomendações.
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
import os

# Configuração de visualização
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

def gerar_dataset_simulado(num_usuarios=1000, num_titulos=50, visualizacoes_min=5, visualizacoes_max=20):
    """
    Gera um dataset simulado de visualizações de streaming.
    
    Args:
        num_usuarios: Número de usuários a serem simulados
        num_titulos: Número de títulos no catálogo
        visualizacoes_min: Mínimo de títulos que um usuário assiste
        visualizacoes_max: Máximo de títulos que um usuário assiste
        
    Returns:
        DataFrame com as transações de visualização
    """
    print("Gerando dataset simulado...")
    
    # Lista de títulos populares para streaming (simulação)
    titulos_populares = [
        "Stranger Things", "The Crown", "Breaking Bad", "Narcos", "Peaky Blinders",
        "Money Heist", "Dark", "The Witcher", "The Mandalorian", "The Queen's Gambit",
        "Bridgerton", "Lupin", "The Umbrella Academy", "You", "The Last Dance",
        "Tiger King", "The Office", "Friends", "Game of Thrones", "Black Mirror",
        "Ozark", "Better Call Saul", "House of Cards", "The Boys", "Westworld",
        "The Handmaid's Tale", "Fargo", "True Detective", "Mindhunter", "Big Little Lies",
        "Succession", "Chernobyl", "When They See Us", "The Good Place", "Fleabag",
        "Killing Eve", "The Marvelous Mrs. Maisel", "Ted Lasso", "The Morning Show", "Squid Game",
        "Loki", "WandaVision", "The Falcon and the Winter Soldier", "The Expanse", "The Walking Dead",
        "Vikings", "The 100", "Lost", "Prison Break", "Dexter"
    ]
    
    # Garantir que temos títulos suficientes
    if len(titulos_populares) < num_titulos:
        for i in range(len(titulos_populares), num_titulos):
            titulos_populares.append(f"Título {i+1}")
    
    # Selecionar apenas o número desejado de títulos
    titulos = titulos_populares[:num_titulos]
    
    # Criar grupos de títulos que tendem a ser assistidos juntos (para simular padrões reais)
    grupos_relacionados = [
        ["Breaking Bad", "Better Call Saul", "Narcos", "Ozark", "Peaky Blinders"],
        ["Stranger Things", "Dark", "Black Mirror", "The Witcher"],
        ["The Crown", "Bridgerton", "The Queen's Gambit", "Downton Abbey"],
        ["Game of Thrones", "The Witcher", "The Mandalorian", "Loki", "WandaVision"],
        ["Friends", "The Office", "The Good Place", "Fleabag", "Ted Lasso"]
    ]
    
    # Definir popularidade relativa dos títulos (alguns são mais populares que outros)
    popularidade = {}
    for titulo in titulos:
        # Títulos mais populares têm maior probabilidade de serem assistidos
        if titulo in titulos_populares[:10]:
            popularidade[titulo] = random.uniform(0.4, 0.7)
        else:
            popularidade[titulo] = random.uniform(0.1, 0.4)
    
    # Gerar dados de visualização
    dados = []
    
    for usuario_id in range(1, num_usuarios + 1):
        # Determinar quantos títulos este usuário assistiu
        num_visualizacoes = random.randint(visualizacoes_min, min(visualizacoes_max, num_titulos))
        
        # Determinar se o usuário tem preferência por algum grupo específico
        if random.random() < 0.7:  # 70% dos usuários têm preferência de gênero
            grupo_preferido = random.choice(grupos_relacionados)
            
            # Adicionar alguns títulos do grupo preferido
            titulos_assistidos = set()
            for titulo in grupo_preferido:
                if titulo in titulos and random.random() < 0.8:  # 80% de chance de assistir cada título do grupo
                    titulos_assistidos.add(titulo)
            
            # Complementar com outros títulos aleatórios até atingir num_visualizacoes
            outros_titulos = [t for t in titulos if t not in titulos_assistidos]
            for titulo in sorted(outros_titulos, key=lambda t: random.random()):
                if len(titulos_assistidos) >= num_visualizacoes:
                    break
                if random.random() < popularidade[titulo]:
                    titulos_assistidos.add(titulo)
        else:
            # Usuários sem preferência clara escolhem com base na popularidade geral
            titulos_assistidos = set()
            for titulo in sorted(titulos, key=lambda t: random.random()):
                if len(titulos_assistidos) >= num_visualizacoes:
                    break
                if random.random() < popularidade[titulo]:
                    titulos_assistidos.add(titulo)
        
        # Garantir que o usuário assistiu a pelo menos um título
        while len(titulos_assistidos) == 0:
            titulo = random.choice(titulos)
            titulos_assistidos.add(titulo)
        
        # Adicionar entradas para cada título assistido por este usuário
        for titulo in titulos_assistidos:
            # Simular uma data de visualização nos últimos 90 dias
            dias_atras = random.randint(1, 90)
            data_visualizacao = datetime.now().replace(
                hour=random.randint(0, 23),
                minute=random.randint(0, 59),
                second=random.randint(0, 59)
            )
            
            dados.append({
                'usuario_id': usuario_id,
                'titulo': titulo,
                'data_visualizacao': data_visualizacao
            })
    
    # Converter para DataFrame
    df = pd.DataFrame(dados)
    print(f"Dataset gerado com {len(df)} registros de visualização de {num_usuarios} usuários.")
    return df

def preparar_dados_para_apriori(df):
    """
    Converte os dados de visualização para o formato adequado para o algoritmo Apriori.
    
    Args:
        df: DataFrame com as colunas 'usuario_id' e 'titulo'
        
    Returns:
        DataFrame no formato one-hot encoding para uso com Apriori
    """
    # Agrupar por usuário para criar "cestas" de visualização
    cestas_visualizacao = df.groupby('usuario_id')['titulo'].apply(list).values.tolist()
    
    # Usar TransactionEncoder para converter para formato one-hot
    te = TransactionEncoder()
    te_ary = te.fit_transform(cestas_visualizacao)
    
    # Criar DataFrame com codificação one-hot
    df_one_hot = pd.DataFrame(te_ary, columns=te.columns_)
    
    return df_one_hot

def executar_apriori(df_one_hot, min_support=0.03, min_confidence=0.3, min_lift=1.5):
    """
    Executa o algoritmo Apriori para encontrar conjuntos frequentes e regras de associação.
    
    Args:
        df_one_hot: DataFrame no formato one-hot para Apriori
        min_support: Suporte mínimo para considerar um conjunto frequente
        min_confidence: Confiança mínima para considerar uma regra de associação
        min_lift: Lift mínimo para filtrar regras de associação
        
    Returns:
        DataFrame com as regras de associação encontradas
    """
    print(f"Executando Apriori (suporte={min_support}, confiança={min_confidence}, lift={min_lift})...")
    
    # Encontrar conjuntos frequentes
    frequent_itemsets = apriori(df_one_hot, min_support=min_support, use_colnames=True)
    
    # Se não houver conjuntos frequentes, retornar DataFrame vazio
    if len(frequent_itemsets) == 0:
        print("Nenhum conjunto frequente encontrado com os parâmetros atuais.")
        return pd.DataFrame()
    
    # Gerar regras de associação
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Filtrar por lift para garantir regras significativas
    filtered_rules = rules[rules['lift'] >= min_lift]
    
    # Ordenar por lift (descendente)
    filtered_rules = filtered_rules.sort_values('lift', ascending=False)
    
    print(f"Encontradas {len(filtered_rules)} regras de associação.")
    return filtered_rules

def formatar_regras(rules):
    """
    Formata as regras de associação para uma linguagem natural.
    
    Args:
        rules: DataFrame com as regras de associação
        
    Returns:
        Lista de strings com as regras formatadas
    """
    regras_formatadas = []
    
    for _, row in rules.iterrows():
        antecedentes = list(row['antecedents'])
        consequentes = list(row['consequents'])
        
        if len(antecedentes) == 1:
            regra = f"Quem assistiu {antecedentes[0]} também pode gostar de {consequentes[0]}."
        else:
            antec_str = " e ".join([f"'{item}'" for item in antecedentes])
            regra = f"Quem assistiu {antec_str} também pode gostar de {consequentes[0]}."
        
        confianca = row['confidence'] * 100
        lift = row['lift']
        
        regra += f" (Confiança: {confianca:.1f}%, Lift: {lift:.2f})"
        regras_formatadas.append(regra)
    
    return regras_formatadas

def visualizar_regras_top(rules, top_n=10):
    """
    Visualiza as melhores regras de associação em um gráfico.
    
    Args:
        rules: DataFrame com as regras de associação
        top_n: Número de regras a serem visualizadas
    """
    if len(rules) == 0:
        print("Sem regras para visualizar.")
        return
    
    # Limitar ao número de regras especificado
    top_rules = rules.head(top_n)
    
    # Criar rótulos para o eixo y (formatando as regras)
    y_labels = []
    for _, row in top_rules.iterrows():
        antecedents = ', '.join(list(row['antecedents']))
        consequents = ', '.join(list(row['consequents']))
        y_labels.append(f"{antecedents} → {consequents}")
    
    # Criar figura
    plt.figure(figsize=(10, 8))
    
    # Plotar gráfico de barras para lift
    ax = sns.barplot(x=top_rules['lift'], y=y_labels)
    
    # Adicionar valores de confiança como texto
    for i, (_, row) in enumerate(top_rules.iterrows()):
        confidence = row['confidence']
        lift = row['lift']
        plt.text(lift + 0.1, i, f"Conf: {confidence:.2f}", va='center')
    
    plt.title('Top Regras de Associação (Ordenadas por Lift)')
    plt.xlabel('Lift')
    plt.tight_layout()
    
    # Salvar a figura
    plt.savefig('top_regras_associacao.png')
    print("Gráfico salvo como 'top_regras_associacao.png'")

def salvar_resultados(regras_formatadas, arquivo_saida='recomendacoes_streaming.txt'):
    """
    Salva as regras formatadas em um arquivo de texto.
    
    Args:
        regras_formatadas: Lista de strings com as regras formatadas
        arquivo_saida: Nome do arquivo para salvar os resultados
    """
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        f.write("Recomendações Geradas por Análise de Associação\n")
        f.write("==============================================\n\n")
        
        for i, regra in enumerate(regras_formatadas, 1):
            f.write(f"{i}. {regra}\n")
    
    print(f"Resultados salvos em '{arquivo_saida}'")

def main():
    """Função principal do programa"""
    print("Análise de Preferências em Catálogo de Streaming com APRIORI")
    print("="*70)
    
    # Gerar dataset simulado de visualizações
    df_visualizacoes = gerar_dataset_simulado(
        num_usuarios=1000, 
        num_titulos=50, 
        visualizacoes_min=5, 
        visualizacoes_max=15
    )
    
    # Mostrar informações do dataset
    print("\nInformações do Dataset:")
    print(f"- Número de usuários: {df_visualizacoes['usuario_id'].nunique()}")
    print(f"- Número de títulos: {df_visualizacoes['titulo'].nunique()}")
    print(f"- Média de visualizações por usuário: {df_visualizacoes.groupby('usuario_id').size().mean():.1f}")
    
    # Títulos mais populares
    top_titulos = df_visualizacoes['titulo'].value_counts().head(10)
    print("\nTop 10 Títulos Mais Populares:")
    for titulo, contagem in top_titulos.items():
        print(f"- {titulo}: {contagem} visualizações")
    
    # Preparar dados para Apriori
    df_one_hot = preparar_dados_para_apriori(df_visualizacoes)
    
    # Executar Apriori com diferentes parâmetros para encontrar boas regras
    min_support = 0.03  # Pelo menos 3% dos usuários têm esse padrão
    min_confidence = 0.3  # 30% de confiança mínima
    min_lift = 1.5  # Lift mínimo para garantir regras significativas
    
    # Tentar ajustar o suporte se necessário para encontrar regras
    regras = executar_apriori(df_one_hot, min_support, min_confidence, min_lift)
    
    if len(regras) == 0:
        print("\nTentando com suporte menor...")
        min_support = 0.02
        regras = executar_apriori(df_one_hot, min_support, min_confidence, min_lift)
    
    if len(regras) == 0:
        print("\nTentando com suporte ainda menor...")
        min_support = 0.01
        regras = executar_apriori(df_one_hot, min_support, min_confidence, min_lift)
    
    # Se encontrou regras, processar e exibir resultados
    if len(regras) > 0:
        # Formatar regras para apresentação
        regras_formatadas = formatar_regras(regras)
        
        # Exibir top regras
        print("\nTop 10 Recomendações Geradas:")
        for i, regra in enumerate(regras_formatadas[:10], 1):
            print(f"{i}. {regra}")
        
        # Visualizar as melhores regras
        visualizar_regras_top(regras)
        
        # Salvar resultados em arquivo
        salvar_resultados(regras_formatadas)
        
    else:
        print("\nNão foi possível encontrar regras de associação significativas com os parâmetros atuais.")
        print("Sugestões:")
        print("- Aumentar o número de usuários no dataset")
        print("- Diminuir os valores de suporte mínimo, confiança ou lift")
        print("- Verificar a distribuição dos dados para garantir que existam padrões detectáveis")

if __name__ == "__main__":
    main() 