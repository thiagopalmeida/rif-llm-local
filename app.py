from typing import Optional
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from pydantic import BaseModel, Field
from typing import List, Optional
# from langchain_core.utils.function_calling import tool_example_to_messages
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_ollama import ChatOllama
import streamlit as st
import pandas as pd
import re
from utils import *
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_selectable_textarea import st_selectable_textarea
import json
from io import StringIO

st.set_page_config(layout="wide", page_title="RIF INTELIGENTE", page_icon="🧠")
st.markdown("# 🧠 Editor Inteligente de Texto com LLM Local")
st.markdown("Transforme textos automaticamente com superpoderes de IA local ✨.")

# Upload do CSV
uploaded_file = st.file_uploader("📄 Faça upload de um arquivo CSV COMUNICAÇÃO", type="csv")
uploaded_file2 = st.file_uploader("📄 Faça upload de um arquivo CSV ENTIDADES", type="csv")

if uploaded_file and uploaded_file2:
    df = pd.read_csv(uploaded_file, encoding="iso-8859-1", delimiter=";", dtype={'Indexador': str})
    df = df[df.CodigoSegmento == 41].reset_index(drop=True)
    coluna_escolhida = 'informacoesAdicionais'
    
    if coluna_escolhida in df.columns:
        lista_comunicacoes = df.idComunicacao.values.tolist()
        
        indexador = df.Indexador.values.tolist()

        df_env = pd.read_csv(uploaded_file2, encoding="iso-8859-1", delimiter=";", dtype={'Indexador': str})

        # para criar lista envolvidos CPF/CNPJ e NOMES
        todos_envolvidos_cpf_cnpj = df_env.cpfCnpjEnvolvido.dropna().astype(str).tolist()
        todos_envolvidos_nomes = df_env.nomeEnvolvido.dropna().tolist()
        # lista única
        todos_envolvidos_cpf_cnpj = list(set(todos_envolvidos_cpf_cnpj))
        todos_envolvidos_nomes = list(set(todos_envolvidos_nomes))


        lista_comunicacoes_editada = []
        for comunica in lista_comunicacoes:
            lista_comunicacoes_editada.append(str(comunica)[:-2])

        comunic_numero = st.select_slider("Escolha a comunicação", options=lista_comunicacoes_editada)
        idx = df.loc[df['idComunicacao'] == float(comunic_numero)].index[0]
        ind = df.at[idx, "Indexador"]
        texto_original = df.at[idx, coluna_escolhida]
        banco_comunicante = df.at[idx, "nomeComunicante"]

        # para selecionar apenas os indexes da comunicação
        df_env = df_env[df_env['Indexador'].isin(indexador)]

        # st.dataframe(df_env) BRONCA AQUI (LEMBRAR DE ESPECIFICAR OS FORMATOS DAS COLUNAS NA IMPORTAÇÃO PARA EVITAR ERROS)

        st.dataframe(df_env[df_env.Indexador == ind])

        indexador_env = df.at[idx, "Indexador"]
        df_env_selecionados = df_env[df_env['Indexador'] == indexador_env]
        # Pegar aqui o primeiro titular da comunicação (apenas 1)
        titular_env = df_env_selecionados[df_env_selecionados.tipoEnvolvido.str.strip().str.lower() == "titular"].iloc[0]

        selecionados_envolvidos = df_env_selecionados["cpfCnpjEnvolvido"].to_list()
        qtd_envolvidos = len(set(selecionados_envolvidos))

        name_file = str(df.at[idx, "idComunicacao"])[:-2]

        # Pré-processamento
        texto_preprocessado = preprocessar_texto_completo(texto_original, todos_envolvidos_nomes, todos_envolvidos_cpf_cnpj)

        # TESTE PARA SELECIONAR TEXTO ---------
        
        # texto = st.text_area("Insira o texto completo:")
        selecao_origem = st_selectable_textarea(value=texto_original, key="selecao_texto_origem", height=300)
        st.text_area("Texto selecionado remetente de recursos:", selecao_origem, height=300, key = "origem_text")
        selecao_destino = st_selectable_textarea(value=texto_original, key="selecao_texto_destino", height=300)
        st.text_area("Texto selecionado beneficiários de recursos:", selecao_destino, height=300, key = "destino_text")
        
        # PREPROCESSAR
        geral = texto_original.replace(selecao_origem, "").replace(selecao_destino, "")
        geral_preprocessado = preprocessar_texto_completo(geral, todos_envolvidos_nomes, todos_envolvidos_cpf_cnpj)
        origem_preprocessado = preprocessar_texto_completo(selecao_origem, todos_envolvidos_nomes, todos_envolvidos_cpf_cnpj)
        destino_preprocessado = preprocessar_texto_completo(selecao_destino, todos_envolvidos_nomes, todos_envolvidos_cpf_cnpj)

        textos_separados_lista = juntar_textos(geral_preprocessado, origem_preprocessado, destino_preprocessado)

        # FINAL DO TESTE -------
        
        
        col1, col2 = st.columns(2)


        with col1:
            st.markdown("#### ✍️ Texto original - Comunicação {} - {} entidades".format(name_file, qtd_envolvidos))
            # st.markdown("#### Comunicação {}".format(name_file))
            # Checkbox para ativar destaque
            destacar = st.checkbox("Destacar entidades conhecidas", value=False)
            valores = encontrar_valores(texto_original)
            entidades = juntar_entidades(todos_envolvidos_nomes, todos_envolvidos_cpf_cnpj, valores)

            if destacar:
                
                html_destacado = destacar_entidades_html(
                    texto_original,
                    entidades,
                )
                st.markdown(html_destacado, unsafe_allow_html=True)
            else:
                st.text_area("Texto original", texto_original, height=500, key="original_text")

        with col2:
            st.markdown("#### 🔁 Texto transformado")

            col1_button, col2_button = st.columns([1, 1])

            # Garante que o estado esteja inicializado
            if "texto_transformado" not in st.session_state:
                st.session_state["texto_transformado"] = ""
            if "mostrar_dialog" not in st.session_state:
                st.session_state["mostrar_dialog"] = False

            with col1_button:
                if st.button("🚀 Transformar"):

                    # texto_transformado = transformar_texto_com_llm(texto_preprocessado)

                    # teste ------
                    texto_transformado = aplicar_llm_por_secao(textos_separados_lista)
                    texto_transformado = limpar_resultado_final(texto_transformado)
                    # fim -------

                    st.session_state["texto_transformado"] = texto_transformado
                
            with col2_button:
                if st.button("🔄 Alternar visualização"):
                    st.session_state["mostrar_dialog"] = True

            # Editor de texto (sempre visível)
            novo_texto = st.text_area("✏️ Edite o texto Markdown abaixo:", st.session_state["texto_transformado"], height=500, key="result_text")
            st.session_state["texto_transformado"] = novo_texto

            col1_button_download, col2_button_download = st.columns([1, 1])
           
            with col1_button_download:
                if novo_texto:
                    st.download_button("💾 Baixar como .md", st.session_state["texto_transformado"], file_name="{}_transformado.md".format(name_file), help="Baixa apenas o texto em Markdown", key="download_md")
            with col2_button_download:
                if novo_texto:
                    # FALTA IMPLEMENTAR                    
                    json_para_baixar = {"comunicacao": "{}".format(name_file),
                                        "texto_bruto_total": "{}".format(texto_original),
                                        "texto_bruto_geral": "{}".format(geral),
                                        "texto_bruto_origem": "{}".format(selecao_origem),
                                        "texto_bruto_destino": "{}".format(selecao_destino),
                                        "texto_transformado_total": "{}".format(novo_texto),
                                        "texto_transformado_geral": "{}".format(str(st.session_state["texto_transformado"]).split('\n\n ------------ \n\n')[0]),
                                        "texto_transformado_origem": "{}".format(str(st.session_state["texto_transformado"]).split('\n\n ------------ \n\n')[1]),
                                        "texto_transformado_destino": "{}".format(str(st.session_state["texto_transformado"]).split('\n\n ------------ \n\n')[2])
                    }
                    json_string = json.dumps(json_para_baixar, ensure_ascii=False, indent=4)

                    # st.json(json_string, expanded=True)
                    
                    st.download_button("💾 Baixar como .json", json_string.encode('utf-8'), file_name="{}_tudo.json".format(name_file), help="Baixa todos os textos brutos e transformados (8 no total)", key="download_json")

        # Se ativado, exibe o expander com markdown renderizado
        if st.session_state["mostrar_dialog"]:
            expandir = st.expander("👁️ Visualização da comunicação em Markdown")
            with expandir:
                st.markdown(st.session_state.get("texto_transformado", ""), unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### 🧮 Tabela editável Origem/Destino dos recursos")

        # TESTE TABELA ------------------

        # Encontrar todos os blocos de tabela markdown
        tabela_origem = re.findall(r"((?:\|.*\n)+)", st.session_state["texto_transformado"].split('\n\n ------------ \n\n')[1])
        tabela_destino = re.findall(r"((?:\|.*\n)+)", st.session_state["texto_transformado"].split('\n\n ------------ \n\n')[2])
        

        df_origem_final = juntar_tabelas_markdown(tabela_origem)
        df_destino_final = juntar_tabelas_markdown(tabela_destino)
        # df_origem_final = pd.concat(dataframes, ignore_index=True)

        df_origem_destino_final = criar_tabela_origem_destino_preenchida(name_file, banco_comunicante, titular_env, df_origem_final, df_destino_final)

        # tratar dados presentes na tabela de maneira inicial
        df_origem_destino_final = limpar_dataframe_valor(df_origem_destino_final)
        df_origem_destino_final["Remetente_CPF_CNPJ"] = df_origem_destino_final["Remetente_CPF_CNPJ"].apply(best_match, args=(df_env["cpfCnpjEnvolvido"].to_list(),))
        df_origem_destino_final["Destinatário_CPF_CNPJ"] = df_origem_destino_final["Destinatário_CPF_CNPJ"].apply(best_match, args=(df_env["cpfCnpjEnvolvido"].to_list(),))
        
        
        st.write(df_origem_destino_final)
        st.download_button(
        "💾 Baixar tabela como CSV",
        df_origem_destino_final.to_csv(index=False),
        file_name="{}_origem_destino.csv".format(name_file),
        key="download_origem_destino")

        # FINAL TESTE ----------------

        # Colunas e configurações
        colunas = ['Comunicação', 'Remetente_Nome', 'Remetente_CPF_CNPJ', 'Remetente_Banco', 'Valor', 'Quantidade', 'Destinatário_Nome', 'Destinatário_CPF_CNPJ', 'Destinatário_Banco']
        valores_padrao = {
            "Comunicação": name_file,
            "Remetente_Nome": "",
            "Remetente_CPF_CNPJ": "",
            "Remetente_Banco": "",
            "Valor": "",
            "Quantidade": "",
            "Destinatário_Nome":"",
            "Destinatário_CPF_CNPJ":"",
            "Destinatário_Banco":""
        }
        if "tabela_editavel" not in st.session_state:
            st.session_state["tabela_editavel"] = pd.DataFrame(columns=colunas)

        tabela = st.session_state["tabela_editavel"].copy()
        tabela.insert(0, "Linha", range(1, len(tabela) + 1))  # adiciona coluna numérica para referência



        # Configurar tabela interativa
        gb = GridOptionsBuilder.from_dataframe(tabela)
        gb.configure_default_column(editable=True, resizable=True)
        gb.configure_selection('multiple', use_checkbox=True)

        gb.configure_column("Linha", header_name="Linha", editable=False, width=70, pinned='left', checkboxSelection=True)

        # Coluna: Comunicação
        gb.configure_column("Comunicação", header_name="Comunicação", editable=True)
        # Coluna: Remetente_Nome (obrigatória)
        gb.configure_column("Remetente_Nome", header_name="Nome Remetente", editable=True, cellEditor="agTextCellEditor")
        # Coluna: Remetente_CPF_CNPJ
        gb.configure_column("Remetente_CPF_CNPJ", header_name="NI Remetente", editable=True)
        # Coluna: Remetente_Banco
        gb.configure_column("Remetente_Banco", header_name="Banco Remetente", editable=True)
         # Coluna: Valor (obrigatória)
        gb.configure_column("Valor", header_name="Valor", editable=True, type=["numericColumn","numberColumnFilter"], cellEditor="agTextCellEditor")
         # Coluna: Quantidade
        gb.configure_column("Quantidade", header_name="Quantidade", editable=True, type=["numericColumn","numberColumnFilter"])
         # Coluna: Destinatário_Nome (obrigatória)
        gb.configure_column("Destinatário_Nome", header_name="Nome Destinatário", editable=True, cellEditor="agTextCellEditor")
         # Coluna: Destinatário_CPF_CNPJ
        gb.configure_column("Destinatário_CPF_CNPJ", header_name="NI Destinatário", editable=True)
         # Coluna: Destinatário_Banco
        gb.configure_column("Destinatário_Banco", header_name="Banco Destinatário", editable=True)


        grid = AgGrid(
            tabela,
            gridOptions=gb.build(),
            editable=True,
            fit_columns_on_grid_load=True,
            theme="alpine",  # pode trocar para "streamlit", "material", "balham"
            update_mode=GridUpdateMode.MODEL_CHANGED,
            allow_unsafe_jscode=True,
            enable_enterprise_modules=False,
            height=400
        )

        # Atualizar dados com edição
        df_editado = pd.DataFrame(grid["data"]).drop(columns=["Linha"], errors="ignore")
        st.session_state["tabela_editavel"] = df_editado

        col1_tabela, col2_tabela, col3_tabela = st.columns([5, 1, 1])

        with col1_tabela:
        
            if st.button("🤯 Preencher automaticamente"):
                nova_linha = df_origem_destino_final
                st.session_state["tabela_editavel"] = pd.concat(
                    [st.session_state["tabela_editavel"], nova_linha], ignore_index=True
                )
                st.rerun()

        with col2_tabela:
            # Adicionar nova linha
            if st.button("➕ Adicionar linhas"):
                
                nova_linha = pd.DataFrame([valores_padrao])
                st.session_state["tabela_editavel"] = pd.concat(
                    [st.session_state["tabela_editavel"], nova_linha], ignore_index=True
                )
                st.rerun()

        # Não tá funcionando ------------- Criar função para apagar toda tabela tb
        with col3_tabela:
            if st.button("➖ Apagar linhas"):
                selecionadas = grid.get("selected_rows", [])
                if isinstance(selecionadas, list) and len(selecionadas) > 0:
                    # Obter índices das linhas selecionadas
                    linhas_para_remover = [row["Linha"] - 1 for row in selecionadas if "Linha" in row]
                    nova_df = st.session_state["tabela_editavel"].drop(index=linhas_para_remover).reset_index(drop=True)
                    st.session_state["tabela_editavel"] = nova_df
                    st.rerun()


    




        # st.session_state["tabela_editavel"] = df_editado

        # Validação simples antes de salvar
        erros = []

        for i, row in df_editado.iterrows():
            if not row["Remetente_Nome"] or not row["Valor"] or not row["Destinatário_Nome"]:
                erros.append(f"Linha {i+1} incompleta (Remetente_Nome, Valor e Destinatário_Nome são obrigatórios)")

        if erros:
            for e in erros:
                st.error(e)
        else:
            # Baixar CSV
            st.download_button(
                "💾 Baixar tabela como CSV",
                pd.DataFrame(df_editado).to_csv(index=False),
                file_name="{}_origem_destino.csv".format(name_file)
        )
    else:
        st.warning("⚠️ Nenhuma coluna de texto encontrada no CSV.")
