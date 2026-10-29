
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Frases utilizadas para treinar a inteligência artificial
frases = [
    "Eu gostei muito desse produto",
    "O atendimento foi excelente",
    "Estou muito feliz com o resultado",
    "Esse filme é maravilhoso",
    "A experiência foi ótima",
    "Adorei o serviço",
    "O produto é muito bom",
    "Estou satisfeito com a compra",
    "O dia está lindo",
    "Foi uma experiência incrível",
    "Eu odiei esse produto",
    "O atendimento foi péssimo",
    "Estou muito triste com o resultado",
    "Esse filme é horrível",
    "A experiência foi ruim",
    "Detestei o serviço",
    "O produto é muito ruim",
    "Estou insatisfeito com a compra",
    "O dia está terrível",
    "Foi uma experiência decepcionante"
]

# Classificações das frases: positivo ou negativo
sentimentos = [
    "Positivo", "Positivo", "Positivo", "Positivo",
    "Positivo", "Positivo", "Positivo", "Positivo",
    "Positivo", "Positivo",
    "Negativo", "Negativo", "Negativo", "Negativo",
    "Negativo", "Negativo", "Negativo", "Negativo",
    "Negativo", "Negativo"
]

# Criando o modelo de Machine Learning
modelo = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)

# Treinando a inteligência artificial
modelo.fit(frases, sentimentos)


def analisar_sentimento(texto):
    """Classifica uma frase como positiva ou negativa."""
    resultado = modelo.predict([texto])[0]
    return resultado


# Programa principal
print("=" * 40)
print(" CLASSIFICADOR DE SENTIMENTOS COM IA")
print("=" * 40)

while True:
    texto = input(
        "\nDigite uma frase (ou 'sair' para encerrar): "
    ).strip()

    if texto.lower() == "sair":
        print("Programa encerrado. Obrigado!")
        break

    if not texto:
        print("Por favor, digite uma frase.")
        continue

    sentimento = analisar_sentimento(texto)

    print(f"Sentimento identificado: {sentimento}")
