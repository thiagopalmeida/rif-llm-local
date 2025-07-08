import requests
import re
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from markupsafe import escape
from html import escape
import unicodedata
from io import StringIO
import pandas as pd
import difflib
from typing import List, Optional
import numpy as np
from rapidfuzz import process, fuzz



# Carrega o modelo do spaCy em português (carregado 1x)
# nlp_pt = spacy.load("pt_core_news_sm")

def normalizar_texto(texto):
    texto = unicodedata.normalize("NFKC", texto)
    texto = texto.replace('\xa0', ' ')  # remove espaço especial
    return texto

def aplicar_mascara_cpf(cpf):
    cpf = re.sub(r"\D", "", cpf)
    return f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}"

def aplicar_mascara_cnpj(cnpj):
    cnpj = re.sub(r"\D", "", cnpj)
    return f"{cnpj[:2]}.{cnpj[2:5]}.{cnpj[5:8]}/{cnpj[8:12]}-{cnpj[12:]}"

# PREPROCESSAMENTO
# 1. Remove espaços e quebras desnecessárias
def limpar_espacos(texto: str) -> str:
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


# 2. Aplica pontuação automática com spaCy
# def pontuar_texto(texto: str) -> str:
#     doc = nlp_pt(texto)
#     sentencas = [sent.text.strip() for sent in doc.sents]
#     return " ".join(sentencas)


# 2. Destaca visualmente entidades úteis como CPF, CNPJ e valores
def destacar_entidades(texto: str) -> str:
    texto = re.sub(r'\b\d{3}\.?\d{3}\.?\d{3}\-?\d{2}\b', r'[CPF:**\g<0>**]', texto)
    texto = re.sub(r'\b\d{2}\.?\d{3}\.?\d{3}\/?\d{4}\-?\d{2}\b', r'[CNPJ:**\g<0>**]', texto)
    texto = re.sub(r'[RU].\s\d+.*?,\d{2}?', lambda x: f"[VALOR:**{x.group()}**]", texto)
    texto = re.sub(r'(\d{1,3}[.,]\d{1,3}?[.,]?\d{1,3}[.,]\d{2})[\s,]', lambda x: f"[VALOR:**{x.group(1)}**]", texto)
    return texto

# # 3. Destaca visualmente entidades úteis como NOMES, CPF, CNPJ e valores
# def destacar_entidades(texto: str, dicionario: dict) -> str:
    
#     texto = re.sub(r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b', r'[CPF: \g<0>]', texto)
#     texto = re.sub(r'\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b', r'[CNPJ: \g<0>]', texto)
#     texto = re.sub(r'[RU].\s\d+.*?,\d{2}?', lambda x: f"[VALOR: {x.group()}]", texto)
#     texto = re.sub(r'\d{1,3}[.,]\d{1,3}?[.,]?\d{1,3}[.,]\d{2}[\s,]', lambda x: f"[VALOR: {x.group()}]", texto)
#     return texto

# 3. Destacar entidades relevantes NOME, CPF, CNPJ
def destacar_entidades_relevantes(texto, nomes=None, cpfs_cnpjs=None):
    if nomes:
        for nome in nomes:
            texto = re.sub(
                re.escape(nome),
                f"[NOME_RELEVANTE:**{nome}**]", 
                texto, 
                flags=re.IGNORECASE
            )
    if cpfs_cnpjs:
        for cpf_cnpj in cpfs_cnpjs:
            len_cpf_cnpj = len(re.sub(r"\D", "", str(cpf_cnpj)))
            if len_cpf_cnpj == 14:
                cnpj = cpf_cnpj 
                texto = re.sub(r"\[CNPJ:\s{}\.?{}\.?{}/?{}-?{}\]".format(cnpj[:2],cnpj[3:6], cnpj[7:10], cnpj[11:15], cnpj[16:]), r'[CNPJ_RELEVANTE**{}**]'.format(aplicar_mascara_cnpj(cnpj)), texto)
                
            elif len_cpf_cnpj == 11:
                cpf = cpf_cnpj
                texto = re.sub(r"\[CPF:\s{}\.?{}\.?{}-?{}\]".format(cpf[:3],cpf[4:7], cpf[8:11], cpf[12:]), r'[CPF_RELEVANTE:**{}**]'.format(aplicar_mascara_cpf(cpf)), texto)
                
    return texto


# # 4. Divide o texto em blocos menores usando LangChain
# def dividir_em_blocos(texto: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list:
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", ".", " "]
#     )
#     return splitter.split_text(texto)

# 4. Pipeline completo
def preprocessar_texto_completo(texto, nomes=None, cpfs_cnpjs=None):
    texto_limpo = limpar_espacos(texto)
    texto_com_entidades = destacar_entidades(texto_limpo)
    texto_com_entidades = destacar_entidades_relevantes(texto_com_entidades, nomes, cpfs_cnpjs)
    # blocos = dividir_em_blocos(texto_com_entidades)
    return texto_com_entidades

# # 5. Pipeline completo
# def preprocessar_texto_completo(texto: str, dicionario_auxiliar: dict = None) -> list:
#     texto_limpo = limpar_espacos(texto)
#     # texto_pontuado = pontuar_texto(texto_limpo)
#     texto_com_entidades = destacar_entidades(texto_limpo)
#     # blocos = dividir_em_blocos(texto_com_entidades)

#     return texto_com_entidades

def encontrar_valores(texto):
    valores = re.findall(r'\d{1,3}[.,]\d{1,3}?[.,]?\d{1,3}[.,]\d{2}[\s,:]', texto)
    return valores


def juntar_entidades(nomes, cpfs_cnpjs, valores):
# Juntar todas as entidades com tipo
    entidades = []

    for nome in nomes:
        entidades.append((nome, "NOME"))

    for cpf_cnpj in cpfs_cnpjs:
        len_cpf_cnpj = len(re.sub(r"\D", "", str(cpf_cnpj)))
        if len_cpf_cnpj == 14:
            entidades.append((aplicar_mascara_cnpj(cpf_cnpj), "CNPJ"))
        elif len_cpf_cnpj == 11:
            entidades.append((aplicar_mascara_cpf(cpf_cnpj), "CPF"))
    for valor in valores:
        entidades.append((valor, "VALOR"))
    return entidades

    # Ordenar por tamanho decrescente (evita matches dentro de outros)
    entidades.sort(key=lambda x: -len(x[0]))

def destacar_entidades_html(texto, entidades, cor_por_tipo=None):
    """
    Destaca entidades no texto com <span> colorido com base nas posições, evitando regex.

    Args:
        texto (str): Texto original.
        entidades (list): Lista de tuplas (termo, tipo), ex: [("João", "NOME"), ("123.456.789-00", "CPF")]
        cor_por_tipo (dict): Dicionário com cores personalizadas por tipo. Ex: {"NOME": "#fdd", "CPF": "#dfd"}

    Returns:
        str: Texto com HTML destacado seguro para uso em st.markdown(..., unsafe_allow_html=True)
    """
    if cor_por_tipo is None:
        cor_por_tipo = {
            "NOME": "#537ba3",
            "CPF": "#bb7e7e",
            "CNPJ": "#429450",
            "VALOR": "#559094",
        }

    def escape_html_extra(texto):
        return escape(texto).replace('$', '&#36;')

    texto_escapado = escape_html_extra(texto)
    lower_text = texto_escapado.lower()

    spans = []
    ocupados = [False] * len(texto_escapado)

    for termo, tipo in sorted(entidades, key=lambda x: len(x[0]), reverse=True):
        termo_escapado = escape(termo)
        termo_lower = termo_escapado.lower()

        inicio = 0
        while True:
            idx = lower_text.find(termo_lower, inicio)
            if idx == -1:
                break

            # Verifica se já está ocupado (evita sobreposição)
            if any(ocupados[idx:idx + len(termo_escapado)]):
                inicio = idx + len(termo_escapado)
                continue

            # Marca como ocupado
            for i in range(idx, idx + len(termo_escapado)):
                ocupados[i] = True

            spans.append((idx, idx + len(termo_escapado), termo_escapado.upper(), tipo))
            inicio = idx + len(termo_escapado)

    # Ordena por posição
    spans.sort()

    # Monta o novo HTML
    resultado = []
    ultimo_fim = 0
    for inicio, fim, trecho, tipo in spans:
        cor = cor_por_tipo.get(tipo, "#eee")
        resultado.append(texto_escapado[ultimo_fim:inicio])
        resultado.append(
            f"<span style='background-color:{cor}; padding:2px 4px; border-radius:4px;' title='{tipo}'>{trecho}</span>"
        )
        ultimo_fim = fim
    resultado.append(texto_escapado[ultimo_fim:])

    return "".join(resultado)





# def destacar_entidades_annotated(texto, nomes=None, cpfs_cnpjs=None):
#     nomes = nomes or []
#     cpfs_cnpjs = cpfs_cnpjs or []

#     # Máscara de cores
#     cores = {
#         "NOME": ("#d2e3fc", "#000"),
#         "CPF": ("#fcf4d2", "#000"),
#         "CNPJ": ("#d2fcd6", "#000")
#     }

#     # Marcações encontradas
#     anotacoes = []

#     # Juntar todas as entidades com tipo
#     entidades = []

#     for nome in nomes:
#         entidades.append((nome, "NOME"))

#     for cpf_cnpj in cpfs_cnpjs:
#         len_cpf_cnpj = len(re.sub(r"\D", "", str(cpf_cnpj)))
#         if len_cpf_cnpj == 14:
#             entidades.append((aplicar_mascara_cnpj(cpf_cnpj), "CNPJ"))
#         elif len_cpf_cnpj == 11:
#             entidades.append((aplicar_mascara_cpf(cpf_cnpj), "CPF"))

#     # Ordenar por tamanho decrescente (evita matches dentro de outros)
#     entidades.sort(key=lambda x: -len(x[0]))

#     # Percorrer o texto e gerar as anotações
#     cursor = 0
#     while cursor < len(texto):
#         match_encontrado = False
#         for entidade, tipo in entidades:
#             if texto[cursor:].lower().startswith(entidade.lower()):
#                 fg, bg = cores[tipo]
#                 anotacoes.append((texto[cursor:cursor+len(entidade)], tipo, bg, fg))
#                 cursor += len(entidade)
#                 match_encontrado = True
#                 break
#         if not match_encontrado:
#             anotacoes.append(texto[cursor])
#             cursor += 1

#     # Renderizar com annotated_text
#     annotated_text(*anotacoes)






# def destacar_entidades_conhecidas_colorido(texto, nomes=None, cpfs_cnpjs=None):
#     texto_escapado = escape(texto)  # Evita quebra do HTML com caracteres especiais

#     if nomes:
#         for nome in nomes:
#             padrao = re.compile(re.escape(nome), re.IGNORECASE)

#             texto_escapado = padrao.sub(
#                 f' <span style="background-color:#ffd54f; padding:2px; border-radius:3px;">{nome}</span> ',
#                 texto_escapado
#             )

#     if cpfs_cnpjs:
#         for cpf_cnpj in cpfs_cnpjs:
#             numeros = re.sub(r"\D", "", cpf_cnpj)
#             if len(numeros) == 14:
#                 cnpj_masc = aplicar_mascara_cnpj(numeros)
#                 texto_escapado = texto_escapado.replace(
#                     numeros,
#                     f' <span style="background-color:#81d4fa; padding:2px; border-radius:3px;">{cnpj_masc}</span> '
#                 )
#             elif len(numeros) == 11:
#                 cpf_masc = aplicar_mascara_cpf(numeros)
#                 texto_escapado = texto_escapado.replace(
#                     numeros,
#                     f' <span style="background-color:#a5d6a7; padding:2px; border-radius:3px;">{cpf_masc}</span> '
#                 )

#     return texto_escapado



#LANGCHAIN
def transformar_texto_com_llm(texto_usuario: str) -> str:
    """
    Usa um modelo LLM local via LangChain para transformar texto desorganizado
    em um texto estruturado em Markdown.
    """

    llm = ChatOllama(model="phi4:latest", temperature=0)
    # llm = ChatOllama(model="qwen3:4b", temperature=0)

    prompt = montar_prompt(texto_usuario)
    resposta = llm.invoke(prompt)
    resposta = resposta.content.replace("```markdown",'').replace("Aqui está o texto formatado em Markdown:", "").replace("```", "").replace("Este formato em Markdown organiza o texto de forma clara e estruturada, facilitando a leitura e compreensão das informações financeiras apresentadas.", "")
    # resposta = resposta.content.split("\n</think>\n")[1]
    return resposta


def montar_prompt(texto):

    # Exemplos de transformação (few-shots)
    text_example1 = """Consta atuar como agente de compras e vendas (autônomo), com renda mensal de R$ 1.200,00.      Entre 02.05.2018 e 06.03.2020 os créditos somaram R$ 1.033.969,28, sendo R$ 760.778,72 por meio de 239 depósitos realizados nas praças de Capinpolis-MG, Maceió-AL, Maceió-AL (Região Portuária), Porto Calvo-AL, Rio Largo-AL, dos quais R$ 680.865,00 constando como efetuados em espécie através de 212 transações, R$ 74.810,00 efetuados em terminais de autoatendimento, 24 transações, e R$ 273.190,26 provenientes de 101 TEDs, DOCs e transferências entre contas. Demonstramos os principais depositantes e remetentes:       VALOR R$ 	QTD	REMETENTE 			CPF/CNPJ 		BANCO 89.600,00 	13	Mesma titularidade			-		Brasil 76.850,18 	55	MMM	22222222/2222-22	Banco (3380/63333) 43.300,00 	5	Geraldo			555555.555-55	Nordeste do Brasil 20.000,00 	1	Jadson		444.444.444-44	Santander       Os débitos, em igual Período, totalizaram R$ 1.032.349,98, dos quais R$ 415.264,66 utilizados para pagamentos diversos, 175 transações, R$ 74.980,00 constando como sacados em espécie, 47 retiradas, R$ 387.476,63 pagos pela compensação de 98 cheques, e R$ 141.829,54 destinados para quitação de 57 TEDs, DOCs, transferências e depósitos em contas. Demonstramos os principais favorecidos:      VALOR R$ 	QTD	FAVORECIDO		CPF/CNPJ 		BANCO 40.283,00	3	FFF	11.111.111/1111-11	Safra 33.495,00 	4	MMM	- 22.222.222/2222-22		Bradesco (3229/103XX) 39.700,00 	4	Mesma titularidade		-		Brasil 27.062,00 	18	Laciel	333.333.333-33	Bradesco (5029/5XXX18)       O cliente relata que os recursos movimentados em conta são de posse de seu irmão, Cicero, CPF 777.777.777-77, o qual atualmente atua como vereador e que também já figurou como procurador da conta em questão. não há informações sobre a motivação.       não podemos desconsiderar que a Movimentação está incompatível com a capacidade financeira presumida do cliente, bem como aparentemente a conta pode ter sido utilizada em benefício de terceiros, apresenta recebimento de recursos de diversas praças, efetuados também em terminais eletrônicos, constam depósitos e saques realizados em espécie, dificultando a identificação da real origem e destino dos recursos."""
    text_example2 = '''Período analisado: 24/05/2019 - 18/05/2020  Trata-se de cliente deste Banco desde 24/03/2011, cadastrado como MOTOTAXISTA, percebendo rendimentos de R$ 1.850,00, residente na cidade de RECIFE/PE.   Contas analisadas:  2332/16.040XX00  2332/510.XXX.039  Resumo de lançamentos a crédito no Período de 24/05/2019 - 18/05/2020 | Total R$ 261.201,16:    84 DEPOSITOS - R$ 141.965,67  180 DOC/TED - R$ 81.491,23  114 CREDITOS DIVERSOS - R$ 37.335,07    1 ESTORNOS - R$ 408,50   24 JUROS POUPANCA - R$ 0,69  Principais remetentes/depositantes identificados:   PPP - 44.444.444/4444-44 ( OUTRAS ATIVIDADES AUXILIARES DOS SERVICOS FINANCEIROS NAO ESPECIFICADAS ANTERIORMENTE ) - 126 lançamento(s) no total de: R$51.278,35  CARLOS - 666.666.666-66 ( MOTOTAXISTA) -  54 lançamento(s) no total de: R$30.212,88  Resumo de lançamentos a débito no Período de 24/05/2019 - 18/05/2020 | Total R$ 253.338,50:   113 CHEQUES - R$ 122.569,82  147 PAGAMENTO TITULO - R$ 88.791,58   12 SAQUES - R$ 21.010,22   20 PAGAMENTO ENERGIA - R$ 5.988,57   12 CARTAO DE CREDITO - R$ 5.832,91   10 TRANSFERENCIAS - R$ 4.863,50   12 IMPOSTOS - R$ 2.568,91    2 COMPRAS - R$ 848,80    7 PAGAMENTO AGUA - R$ 420,74   16 TARIFAS - R$ 344,01   12 SEGUROS - R$ 99,44  Principais destinatários de recursos identificados:   CICERO- 777.777.777-77 ( ESTUDANTE | PEP ) -   1 lançamento(s) no total de: R$1.228,00  KKK - 77.777.777/7777-77 ( COMERCIO ATACADISTA DE PRODUTOS DE HIGIENE, LIMPEZA E CONSERVACAO DOMICILIAR ) -   3 lançamento(s) no total de: R$1.225,50  LAIZE - 000.000.000-00 ( A - ADMINISTRADORA ) -   2 lançamento(s) no total de: R$810,00  GUSTAVO - 999.999.999-99 ( COMERCIANTE - COMERCIARIO ) -   1 lançamento(s) no total de: R$800,00  FERNANDA - 88.888.888/8888-88 ( TRANSPORTE RODOVIARIO DE PRODUTOS PERIGOSOS ) -   1 lançamento(s) no total de: R$500,00  DIOGO - 888.888.888-88 ( ASSISTENTE TECNICO - AGENTE ADMINISTRATIVO ) -   2 lançamento(s) no total de: R$300,00  Segundo informações, o analisado foi orientado para abertura de conta PJ porém não aderiu. Movimentação trata-se de faturamento de estabelecimento comercial vinculado à géneros alimentícios. Os principais créditos foram depósitos em espécie realizados em terminais de auto atendimento dos quais não pudemos identificar a origem.   O analisado pagou títulos de forma habitual em nome da empresa M M DOS - 99.999.999/9999-99 - COMERCIO VAREJISTA DE MERCADORIAS EM GERAL, em favor de empresas do ramo de comercio atacadista de produtos alimenticios em geral, como as empresas VVV - 66.666.666/6666-66, sem cadastro no banco e, as empresas MMM - 22.222.222/2222-22, DDD - 00.000.000/0000-00, RIBEIRO - 55.555.555/5555-55 entre outras.  Movimentação incompatível com a atividade economica do analisado e seus relacionados. Suspeitamos de Movimentação em favor de terceiros ou sonegação fiscal.  Considerando que não foram encontradas justificativas para a Movimentação financeira, comunicamos pela possibilidade de constituir-se em indícios do crime de lavagem de dinheiro, ou com ele relacionar-se.'''
    
    text_example1_result = '''
    # Resumo da Análise Financeira 

    ## Informações do Cliente 
    - **Atividade Profissional:** Agente de compras e vendas (autônomo) 
    - **Renda Mensal:** R$ 1.200,00

    ## Movimentações Financeiras entre 02/05/2018 e 06/03/2020

    ### Créditos
    **Total de Créditos:** R$ 1.033.969,28

    **Detalhamento:**
    - **Depósitos em Espécie:** R$ 680.865,00 (212 transações)
    - **Terminais de Autoatendimento:** R$ 74.810,00 (24 transações)
    - **TEDs, DOCs e Transferências entre Contas:** R$ 273.190,26 (101 transações)

    ### Principais Depositantes

    | Valor (R$)   | Quantidade | Nome               | CPF/CNPJ         | Banco              |
    | ------------ | ---------- | ------------------ | ---------------- | ------------------ |
    | R$ 89.600,00 | 13         | Mesma titularidade | -                | Brasil             |
    | R$ 76.850,18 | 55         | MMM                | 22222222/2222-22 | Banco (3380/63333) |
    | R$ 43.300,00 | 5          | Geraldo            | 555555.555-55    | Nordeste do Brasil |
    | R$ 20.000,00 | 1          | Jadson             | 444.444.444-44   | Santander          |

    ### Débitos 
    **Total de Débitos:** R$ 1.032.349,98

    **Detalhamento:**
    - **Pagamentos Diversos:** R$ 415.264,66 (175 transações)
    - **Saque em Espécie:** R$ 74.980,00 (47 retiradas)
    - **Compensação de Cheques:** R$ 387.476,63 (98 cheques)
    - **TEDs, DOCs e Transferências entre Contas:** R$ 141.829,54 (57 transações)

    ### Principais Favorecidos 

    | Valor (R$)   | Quantidade | Nome               | CPF/CNPJ           | Banco                  |
    | ------------ | ---------- | ------------------ | ------------------ | ---------------------- |
    | 40.283,00    | 3          | FFF                | 11.111.111/1111-11 | Safra                  |
    | 33.495,00    | 4          | MMM                | 22.222.222/2222-22 | Bradesco (3229/103XX)  |
    | 39.700,00    | 4          | Mesma titularidade | -                  | Brasil                 |
    | 27.062,00    | 18         | Laciel             | 333.333.333-33     | Bradesco (5029/5XXX18) |

    ## Observações
    - O cliente afirma que os recursos movimentados na conta são de posse do seu irmão, Cicero, CPF: 777.777.777-77, atualmente vereador e ex-procurador da conta.
    - A movimentação financeira está incompatível com a capacidade financeira presumida do cliente.
    - Há indícios de que a conta pode ter sido utilizada em benefício de terceiros.
    - Os depósitos e saques realizados em espécie dificultam a identificação da origem e destino dos recursos.
    '''

    text_example2_result = '''
    # Resumo da Análise Financeira

    ## Período Analisado: 24/05/2019 - 18/05/2020

    ### Informações do Cliente:
    - **Cliente desde:** 24/03/2011
    - **Profissão:** MOTOTAXISTA
    - **Rendimento declarado:** R$ 1.850,00
    - **Residência:** RECIFE/PE

    ### Contas Analisadas:
    - 2332/16.040XX00
    - 2332/510.XXX.039

    ### Resumo de Lançamentos a Crédito (Total R$ 261.201,16)
    - **Depósitos:** R$ 141.965,67 (84 transações)
    - **DOC/TED:** R$ 81.491,23 (180 transações)
    - **Créditos Diversos:** R$ 37.335,07 (114 transações)
    - **Estornos:** R$ 408,50 (1 transação)
    - **Juros Poupança:** R$ 0,69 (24 transações)

    ### Principais Remetentes/Depositantes Identificados:

    | Valor (R$)  | Quantidade | Nome      | CPF/CNPJ           | Banco  |
    | ----------- | ---------- | --------- | ------------------ | ------ |
    | 51.278,35   | 126        | PPP       | 44.444.444/4444-44 | -      |
    | 30.212,88   | 54         | CARLOS    | 666.666.666-66     | -      |

    ### Resumo de Lançamentos a Débito (Total R$ 253.338,50)
    - **Cheques:** R$ 122.569,82 (113 transações)
    - **Pagamento de Títulos:** R$ 88.791,58 (147 transações)
    - **Saques:** R$ 21.010,22 (12 transações)
    - **Pagamento Energia:** R$ 5.988,57 (20 transações)
    - **Cartão de Crédito:** R$ 5.832,91 (12 transações)
    - **Tranferências:** R$ 4.863,50 (10 transações)
    - **Impostos:** R$ 2.568,91 (12 transações)
    - **Compras:** R$ 848,80  (2 transações)
    - **Pagamento Água:** R$ 420,74 (7 transações)
    - **Tarifas:** R$ 344,01 (16 transações)
    - **Seguros:** R$ 99,44 (12 transações)

    ### Principais Destinatários Identificados:

    | Valor (R$) | Quantidade | Nome         | CPF/CNPJ           | Banco  |
    | ---------- | ---------- | ------------ | ------------------ | ------ |
    | 1.228,00   | 1          | CICERO       | 777.777.777-77     |  -     |
    | 1.225,50   | 3          | KKK          | 77.777.777/7777-77 |  -     |
    | 810,00     | 2          | LAIZE        | 000.000.000-00     |  -     |
    | 800,00     | 1          | GUSTAVO      | 999.999.999-99     |  -     |
    | 500,00     | 1          | FERNANDA     | 88.888.888/8888-88 |  -     |
    | 300,00     | 2          | DIOGO        | 888.888.888-88     |  -     |

    ## Observações e Suspeitas
    - O analisado foi orientado para abrir uma conta PJ, mas não aderiu.
    - A movimentação financeira parece estar relacionada ao faturamento de um estabelecimento comercial vinculado a gêneros alimentícios.
    - Os principais créditos foram depósitos em espécie realizados em terminais de autoatendimento, cuja origem não foi identificada.
    - O analisado pagou títulos habitualmente em nome da empresa M M DOS e outras empresas do ramo de comércio atacadista de produtos alimentícios.
    - A movimentação financeira é incompatível com a atividade econômica declarada pelo cliente, levantando suspeitas de movimentação em favor de terceiros ou sonegação fiscal.

    ## Conclusão
    Considerando que não foram encontradas justificativas para a movimentação financeira observada, comunicamos a possibilidade de constituir-se em indícios do crime de lavagem de dinheiro, ou com ele relacionar-se.
    '''

    exemplos = [
    {"input": preprocessar_texto_completo(text_example1), "output": text_example1_result},
    {"input": preprocessar_texto_completo(text_example2), "output": text_example2_result},
    ]

    prompt = """Você receberá um texto com dados financeiros brutos e destacados por categorias dentro de colchetes, como: [VALOR:**...**], [CPF:**...**], [CNPJ:**...**], [CPF_RELEVANTE:**...**], [CNPJ_RELEVANTE:**...**], [NOME_RELEVANTE:**...**].

    Sua tarefa é reestruturar esse conteúdo de forma organizada, clara e em formato Markdown, sem alterar os valores informados, sem inventar dados, e mantendo a veracidade das informações.
    
    Siga as instruções abaixo:
    
    1. Use títulos e subtítulos com `#`, `##`, `###` para separar seções como "Informações do Cliente", "Resumo de Lançamentos", "Principais Depositantes", "Principais Destinatários", etc.
    2. Liste os principais dados financeiros em **listas ou tabelas** quando apropriado (ex: nome, CPF/CNPJ, valor, quantidade de lançamentos, banco).
    3. Mantenha a ordem dos dados conforme apresentados no texto original.
    4. Não inclua informações que não estejam no texto.
    5. Use as informações destacadas para estruturar os dados com clareza, principalmente os marcados como RELEVANTE.
    6. Organize as seções finais como "Observações e Suspeitas" e "Conclusão", caso essas informações estejam presentes.
        
        Veja os exemplos abaixo:\n\n"""
    for ex in exemplos:
        prompt += f"Texto original:\n{ex['input']}\n\nTexto formatado em Markdown:\n{ex['output']}\n\n"
    
    prompt += "=== FIM DOS EXEMPLOS ==="
    
    prompt += f"Texto original:\n{texto}\n\nTexto formatado em Markdown:\n"
    return prompt

def salvar_markdown(texto: str) -> bytes:
    return texto.encode("utf-8")

# Função para aplicar máscara visual do CPF
def formatar_cpf_cnpj(cpf_cnpj_str):
    cpf_cnpj = re.sub(r"\D", "", str(cpf_cnpj_str))  # remove tudo que não é número
    if len(cpf_cnpj) == 11:
        return f"{cpf_cnpj[:3]}.{cpf_cnpj[3:6]}.{cpf_cnpj[6:9]}-{cpf_cnpj[9:]}"
    elif len(cpf_cnpj) == 14:
        return f"{cpf_cnpj[:2]}.{cpf_cnpj[2:5]}.{cpf_cnpj[5:8]}/{cpf_cnpj[8:12]}-{cpf_cnpj[12:]}"
    return cpf_cnpj  # se não tiver 11 ou 14, retorna sem formatação


# Função para juntar em uma lista os textos selecionados na seguintes ordem (geral, origem, destino)
def juntar_textos(geral, selecao1, selecao2):
    textos_lista = []
    textos_lista.append({"geral": geral})
    textos_lista.append({"origem": selecao1})
    textos_lista.append({"destino": selecao2})
    return textos_lista

# Função para aplicar modelo LLM por seção com prompt especializado
def aplicar_llm_por_secao(textos_lista):
    resultado_final = []
    for texto_key_value in textos_lista:
        for key, value in texto_key_value.items():
            tipo_secao = key
            texto_secao = value

            # Dividir se for muito longo
            splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200, separators=["[NOME", "[CPF", "[CNPJ"])
            trechos = splitter.split_text(texto_secao)
            resultados = []
            if tipo_secao == "geral":
                    prompt = montar_prompt(texto_secao)
            for trecho in trechos:
                # if tipo_secao == "geral":
                #     prompt = montar_prompt(trecho)
            
                if tipo_secao == "origem":
                    prompt = """
            
                    Extraia os seguintes dados de cada entidade presente no texto abaixo:
            
                    - Valor total movimentado
                    - Quantidade de lançamentos
                    - Nome
                    - CPF ou CNPJ
                    - Banco (quando houver)
                    
                    O texto está anotado com marcadores para facilitar a identificação, como [VALOR:**...**], [NOME_RELEVANTE:**...**], [CPF:**...**], [CNPJ:**...**]. Quando o nome não estiver marcado como NOME_RELEVANTE, use o nome mais próximo do CPF ou CNPJ. Ignore duplicações.
                    
                    A saída só deve constar a tabela gerada no formato solicitado e nada mais.
                    
                    A saída deve estar em formato markdown de tabela com as colunas:
                    
                    | Valor | Qtd. Lançamentos | Nome | CPF/CNPJ | Banco |
                    """
                    
                    prompt += f"Texto original:\n{trecho}"
            
                elif tipo_secao == "destino":
                    prompt = """
            
                    Extraia os seguintes dados de cada entidade presente no texto abaixo:
            
                    - Valor total movimentado
                    - Quantidade de lançamentos
                    - Nome
                    - CPF ou CNPJ
                    - Banco (quando houver)
                    
                    O texto está anotado com marcadores para facilitar a identificação, como [VALOR:**...**], [NOME_RELEVANTE:**...**], [CPF:**...**], [CNPJ:**...**]. Quando o nome não estiver marcado como NOME_RELEVANTE, use o nome mais próximo do CPF ou CNPJ. Ignore duplicações.
                    
                    A saída só deve constar a tabela gerada no formato solicitado e nada mais.
                    
                    A saída deve estar em formato markdown de tabela com as colunas:
                    
                    | Valor | Qtd. Lançamentos | Nome | CPF/CNPJ | Banco |
                    """
                    
                    prompt += f"Texto original:\n{trecho}"

                llm = ChatOllama(model="phi4:latest", temperature=0)
                # llm = ChatOllama(model="qwen3:4b", temperature=0)
                resultado = llm.invoke(prompt)
                resultado = resultado.content.replace("```markdown",'').replace("Aqui está o texto formatado em Markdown:", "").replace("```", "").replace("Este formato em Markdown organiza o texto de forma clara e estruturada, facilitando a leitura e compreensão das informações financeiras apresentadas.", "")
                resultados.append(resultado.strip())
        resultado_final.append('\n'.join(resultados))

    return '\n\n ------------ \n\n'.join(resultado_final)

def limpar_resultado_final(texto):
    """
    Limpa e organiza um texto markdown removendo padrões indesejados e caracteres especiais.

    Parâmetros:
        texto (str): O texto original bagunçado.

    Retorna:
        str: O texto limpo e organizado.
    """
    # Substitui padrões específicos
    padroes_remover = [
        re.escape("[['"), re.escape("']], ['"),
        re.escape("'], ['"), re.escape("', '"),
        re.escape("']]"), re.escape("```"), re.escape("$"), re.escape("markdown")
    ]

    # Remove todos os padrões indesejados
    for padrao in padroes_remover:
        texto = re.sub(padrao, '', texto)

    # Limpa excesso de espaços e quebras de linha desnecessárias
    texto = re.sub(r"\n{3,}", "\n\n", texto)       # No máximo 2 quebras
    texto = re.sub(r"[ \t]{2,}", " ", texto)       # Reduz espaços múltiplos
    texto = texto.strip()
    return texto

def juntar_tabelas_markdown(markdown_tables):
    """
    Processa uma lista de strings, onde cada string é uma tabela em formato Markdown,
    e as consolida em um único DataFrame do Pandas.

    A função é robusta para tabelas com múltiplas linhas, uma única linha de dados
    ou mesmo nenhuma linha de dados.

    Args:
        markdown_tables (list): Uma lista de strings, cada uma contendo uma tabela Markdown.

    Returns:
        pd.DataFrame: Um DataFrame consolidado com os dados de todas as tabelas válidas.
                      Retorna um DataFrame vazio se nenhuma tabela válida for encontrada.
    """
    lista_dataframes = []

    for tabela in markdown_tables:
        # 1. Limpeza inicial e separação de linhas, removendo linhas totalmente em branco.
        linhas = [linha.strip() for linha in tabela.strip().split("\n") if linha.strip()]
        
        # 2. Remover a linha separadora (ex: |---|---|, |:--:|--|)
        # A expressão regular agora é mais flexível para incluir os dois pontos usados em alinhamento.
        linhas_sem_separador = [
            linha for linha in linhas 
            if not re.fullmatch(r"\|?[-| :]+\|?", linha.strip())
        ]

        # 3. Validar se temos pelo menos um cabeçalho. Se não, não é uma tabela válida.
        if len(linhas_sem_separador) < 1:
            continue

        # 4. O cabeçalho é a primeira linha. As demais são dados.
        linha_cabecalho = linhas_sem_separador[0]
        linhas_de_dados = linhas_sem_separador[1:]

        # 5. Processar o cabeçalho para obter os nomes das colunas.
        #    - .strip('|') remove os pipes das extremidades.
        #    - .split('|') divide a string nas colunas.
        #    - A list comprehension limpa os espaços em branco de cada nome de coluna.
        nomes_colunas = [col.strip() for col in linha_cabecalho.strip('|').split('|')]

        # 6. Processar cada linha de dados.
        dados_processados = []
        for linha_dado in linhas_de_dados:
            # O mesmo processo do cabeçalho é aplicado a cada linha de dados.
            celulas = [cel.strip() for cel in linha_dado.strip('|').split('|')]
            
            # Garantir que a linha de dados tenha o mesmo número de colunas que o cabeçalho
            # ajuda a evitar erros com linhas mal formatadas.
            if len(celulas) == len(nomes_colunas):
                dados_processados.append(celulas)

        # 7. Criar o DataFrame apenas se tivermos um cabeçalho e dados válidos.
        #    Isso lida corretamente com o caso de tabelas que só têm cabeçalho.
        if nomes_colunas and dados_processados:
            df = pd.DataFrame(dados_processados, columns=nomes_colunas)
            lista_dataframes.append(df)

    # 8. Concatenar todos os DataFrames resultantes em um só.
    if not lista_dataframes:
        return pd.DataFrame() # Retorna um DataFrame vazio se nenhuma tabela foi processada.

    df_final = pd.concat(lista_dataframes, ignore_index=True)
    return df_final

def criar_tabela_origem_destino_preenchida(comunicacao, banco_comunicante, titular, tabela_origem, tabela_destino):
    """
    Estrutura e consolida os dados de transações de origem e destino em um único DataFrame padronizado.

    Args:
        comunicacao (str): Identificador da comunicação.
        banco_comunicante (str): Nome do banco que está comunicando a operação.
        titular (pd.Series): Uma série contendo os dados do titular da conta que fez a comunicação.
                             Espera-se que contenha informações como CPF/CNPJ, Nome, Agência e Conta.
        tabela_origem (pd.DataFrame): DataFrame com os dados das transações de origem.
        tabela_destino (pd.DataFrame): DataFrame com os dados das transações de destino.

    Returns:
        pd.DataFrame: Um DataFrame consolidado com as colunas padronizadas.
                      Retorna um DataFrame vazio se as tabelas de entrada estiverem vazias.
    """
    # Se ambas as tabelas de entrada estiverem vazias, não há o que fazer.
    if tabela_origem.empty and tabela_destino.empty:
        return pd.DataFrame()

    # --- Processamento da Tabela de Origem ---
    df_origem = pd.DataFrame()
    if not tabela_origem.empty:
        # Renomeia as colunas da tabela de origem para nomes temporários e claros
        # com base na sua lógica original de .iloc.
        # Ex: coluna 0 era 'Valor', coluna 1 era 'Quantidade', etc.
        mapeamento_origem = {
            tabela_origem.columns[0]: 'Valor',
            tabela_origem.columns[1]: 'Quantidade',
            tabela_origem.columns[2]: 'Remetente_Nome',
            tabela_origem.columns[3]: 'Remetente_CPF_CNPJ',
            tabela_origem.columns[4]: 'Remetente_Banco'
        }
        df_origem = tabela_origem.rename(columns=mapeamento_origem).copy()

        # Adiciona as colunas de metadados
        df_origem['Comunicação'] = comunicacao
        df_origem['Destinatário_Nome'] = titular.iloc[2]
        df_origem['Destinatário_CPF_CNPJ'] = titular.iloc[1]
        df_origem['Destinatário_Banco'] = f"{banco_comunicante} - {titular.iloc[4]} - {titular.iloc[5]}"

    # --- Processamento da Tabela de Destino ---
    df_destino = pd.DataFrame()
    if not tabela_destino.empty:
        # Renomeia as colunas da tabela de destino para nomes claros.
        mapeamento_destino = {
            tabela_destino.columns[0]: 'Valor',
            tabela_destino.columns[1]: 'Quantidade',
            tabela_destino.columns[2]: 'Destinatário_Nome',
            tabela_destino.columns[3]: 'Destinatário_CPF_CNPJ',
            tabela_destino.columns[4]: 'Destinatário_Banco'
        }
        df_destino = tabela_destino.rename(columns=mapeamento_destino).copy()

        # Adiciona as colunas de metadados
        df_destino['Comunicação'] = comunicacao
        df_destino['Remetente_Nome'] = titular.iloc[2]
        df_destino['Remetente_CPF_CNPJ'] = titular.iloc[1]
        df_destino['Remetente_Banco'] = f"{banco_comunicante} - {titular.iloc[4]} - {titular.iloc[5]}"

    # --- Consolidação Final ---
    df_final = pd.concat([df_origem, df_destino], ignore_index=True)

    # Define a ordem final e garante que todas as colunas existam, preenchendo com NaN se necessário.
    colunas_finais = [
        'Comunicação', 'Remetente_Nome', 'Remetente_CPF_CNPJ', 'Remetente_Banco',
        'Valor', 'Quantidade', 'Destinatário_Nome', 'Destinatário_CPF_CNPJ', 'Destinatário_Banco'
    ]
    
    # Reindexar garante a ordem correta e a presença de todas as colunas
    df_final = df_final.reindex(columns=colunas_finais)

    return df_final


def clean_number(cnpj_cpf: str) -> str:
    """Remove tudo que não for número."""
    return re.sub(r'\D', '', cnpj_cpf)

def detect_type(number: str) -> Optional[str]:
    """Identifica se é CPF ou CNPJ com base no número de dígitos."""
    if len(number) == 11 or len(number) == 10:
        return 'cpf'
    elif len(number) == 14 or len(number) == 13:
        return 'cnpj'
    return None

def best_match(input_dirty: str, valid_list: List[str], similarity_threshold: float = 0.9) -> str:
    """
    Compara um CPF ou CNPJ sujo com uma lista de CPFs ou CNPJs e retorna o mais próximo.
    
    Args:
        input_dirty: O CPF ou CNPJ sujo a ser comparado.
        valid_list: Lista de CPFs/CNPJs válidos (podem estar sujos ou limpos).
        similarity_threshold: Valor mínimo (0 a 1) de similaridade para aceitar substituição.

    Returns:
        O melhor CPF/CNPJ da lista (se similaridade >= limiar), senão o valor original limpo.
    """
    input_clean = clean_number(input_dirty)
    input_type = detect_type(input_clean)
    if not input_type:
        return input_clean  # Retorna como está se não for CPF nem CNPJ

    # Normaliza e filtra lista com base no tipo
    valid_cleaned = [
        clean_number(v) for v in valid_list if detect_type(clean_number(v)) == input_type
    ]

    if not valid_cleaned:
        return input_dirty  # Nenhum do mesmo tipo na lista

    # Calcular similaridade
    matches = difflib.get_close_matches(input_clean, valid_cleaned, n=1, cutoff=similarity_threshold)

    if matches:
        if input_type == "cpf":
            return aplicar_mascara_cpf(matches[0])
        elif input_type == "cnpj":
            return aplicar_mascara_cnpj(matches[0])
    else:
        return input_clean


# def limpar_dataframe_valor(df):
#     # Remove espaços extras em colunas e valores
#     df.columns = df.columns.str.strip()
#     df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

#     # Passar td para upper
#     df['Remetente_Nome'] = df['Remetente_Nome'].str.upper()
#     df['Remetente_Banco'] = df['Remetente_Banco'].str.upper()
#     df['Destinatário_Nome'] = df['Destinatário_Nome'].str.upper()
#     df['Destinatário_Banco'] = df['Destinatário_Banco'].str.upper()

#     # Converte coluna "Valor" para número
#     df["Valor"] = (
#         df["Valor"]
#         .astype(str)
#         .str.replace(".", "", regex=False)
#         .str.replace(",", ".", regex=False)
#         .str.replace("R", "", regex=False)
#         .str.strip()
#         .replace("[^\.\d]*", "", regex=True)
#         .replace("", np.nan)
#         .astype(float)
#     )

#     return df

def juntar_csvs(arquivos_csv):
    """
    Recebe uma lista de objetos UploadedFile do Streamlit e os concatena em um único DataFrame.
    """
    # Passamos cada objeto 'file' diretamente para o pd.read_csv
    dfs = [pd.read_csv(file) for file in arquivos_csv]
    return pd.concat(dfs, ignore_index=True)

def limpar_valor(valor):
    if not isinstance(valor, str):
        valor = str(valor)
    
    # Remove 'R$', 'R', espaços e outros símbolos não numéricos comuns
    valor = re.sub(r'[^\d,.-]', '', valor)
    
    # Se há vírgula e ponto, decidir quem é milhar e quem é decimal
    if ',' in valor and '.' in valor:
        if valor.find(',') < valor.find('.'):
            # Ex: 1,352,805.0 (vírgula = milhar, ponto = decimal)
            valor = valor.replace(',', '')
        else:
            # Ex: 437.839,00 (ponto = milhar, vírgula = decimal)
            valor = valor.replace('.', '').replace(',', '.')
    elif ',' in valor:
        # Assume que vírgula é separador decimal (estilo brasileiro)
        valor = valor.replace(',', '.')
    else:
        # Só ponto ou nenhum: assume ponto como decimal
        pass
    if valor == "":
        valor = np.nan
    
    try:
        return float(valor)
    except ValueError:
        raise ValueError(f"Valor inválido para conversão: {valor}")
        

def limpar_dataframe_ori_dest(df):
    # Remove espaços extras em colunas e valores
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Remove prefixos como "CNPJ:" e espaços nos CPFs/CNPJs
    df["Remetente_CPF_CNPJ"] = df["Remetente_CPF_CNPJ"].str.replace("CNPJ:", "", regex=False).str.strip()
    df["Destinatário_CPF_CNPJ"] = df["Destinatário_CPF_CNPJ"].str.replace("CNPJ:", "", regex=False).str.strip()
    df["Remetente_CPF_CNPJ"] = df["Remetente_CPF_CNPJ"].str.replace("CPF:", "", regex=False).str.strip()
    df["Destinatário_CPF_CNPJ"] = df["Destinatário_CPF_CNPJ"].str.replace("CPF:", "", regex=False).str.strip()

    # Remove linhas com CPF/CNPJ ausente
    df = df.dropna(subset=["Remetente_CPF_CNPJ", "Destinatário_CPF_CNPJ"])

    # Converte coluna "Valor" para número
    df["valor"] = df["Valor"].apply(limpar_valor)
    # df["Valor"] = (
    #     df["Valor"]
    #     .astype(str)
    #     .str.replace(".", "", regex=False)
    #     .str.replace(",", ".", regex=False)
    #     .str.replace("R", "", regex=False)
    #     .str.strip()
    #     .replace("[^\.\d]*", "", regex=True)
    #     .replace("", np.nan)
    #     .astype(float)
    # )

    # Substitui valores NaN em colunas textuais por "Desconhecido"
    for col in ["Remetente_Nome", "Remetente_Banco", "Destinatário_Nome", "Destinatário_Banco", "Comunicação"]:
        df[col] = df[col].fillna("Desconhecido")

    # Garante que não haja valores nulos em campos importantes
    df = df.dropna(subset=["Valor"])

    # Padronizar UPPER nome
    df["Remetente_Nome"] = df["Remetente_Nome"].str.upper()
    df["Destinatário_Nome"] = df["Destinatário_Nome"].str.upper()
    df["Remetente_Banco"] = df["Remetente_Banco"].str.upper()
    df["Destinatário_Banco"] = df["Destinatário_Banco"].str.upper()

    return df

def len_CPF_CNPJ(ni):
    ni = re.sub(r"\D", "", ni)
    return len(ni)

def limpar_documento(doc):
    return re.sub(r'\D', '', str(doc))

def corrigir_doc(doc_sujo, referencia):
    correspondencia = process.extractOne(
        query=doc_sujo,
        choices=referencia,
        scorer=fuzz.ratio,  # ou fuzz.partial_ratio para tolerância maior
        score_cutoff=80     # pode ajustar esse limiar
    )
    return correspondencia[0] if correspondencia else None

def aplicar_mascara(ni):
    if len(ni) == 14:
        cnpj = ni 
        return f"{cnpj[:2]}.{cnpj[2:5]}.{cnpj[5:8]}/{cnpj[8:12]}-{cnpj[12:]}"
        
    elif len(ni) == 11:
        cpf = ni
        return f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}"