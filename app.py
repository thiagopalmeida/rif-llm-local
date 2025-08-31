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

# Define as colunas esperadas para cada arquivo
colunas_esperadas_comunic = ['Indexador', 'idComunicacao', 'NumeroOcorrenciaBC', 'Data_do_Recebimento', 'Data_da_operacao', 'DataFimFato', 'cpfCnpjComunicante', 'nomeComunicante', 'CidadeAgencia', 'UFAgencia', 'NomeAgencia', 'NumeroAgencia', 'informacoesAdicionais', 'CampoA', 'CampoB', 'CampoC', 'CampoD', 'CampoE', 'CodigoSegmento']
colunas_esperadas_entidades = ['Indexador', 'cpfCnpjEnvolvido', 'nomeEnvolvido', 'tipoEnvolvido', 'agenciaEnvolvido', 'contaEnvolvido', 'DataAberturaConta', 'DataAtualizacaoConta', 'bitPepCitado', 'bitPessoaObrigadaCitado', 'intServidorCitado']

# Upload do CSV
uploaded_files = st.file_uploader("📄 Faça upload de um arquivo CSV COMUNICAÇÃO", type="csv", accept_multiple_files=True)
uploaded_files2 = st.file_uploader("📄 Faça upload de um arquivo CSV ENVOLVIDOS", type="csv", accept_multiple_files=True)

df = None
df_env = None

lista_de_dfs = []
for uploaded_file in uploaded_files:
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding="iso-8859-1", delimiter=";", dtype={'Indexador': str, 'cpfCnpjComunicante': str, 'CodigoSegmento': str})
            
            # Validação das colunas do arquivo COMUNICAÇÃO
            colunas_faltando_comunicacao = [col for col in colunas_esperadas_comunic if col not in df.columns]
            
            if colunas_faltando_comunicacao:
                st.error(
                    f"🚨 **Erro no arquivo COMUNICAÇÃO!** As seguintes colunas estão faltando: "
                    f"{', '.join(colunas_faltando_comunicacao)}. "
                    "Por favor, verifique se o arquivo está no formato correto e tente novamente."
                )
                df = None # Define como None para evitar processamento com dados incompletos
            else:
                st.success("✅ Arquivo COMUNICAÇÃO carregado e validado com sucesso!")
            match = re.search(r'(\d+)', uploaded_file.name)
            if match:
                rif_num = match.group(1)
                df['rif_num'] = rif_num
            else:
                # Se não encontrar número, usa o nome do arquivo como fallback
                df['rif_num'] = uploaded_file.name

        except Exception as e:
            st.error(f"Ocorreu um erro ao carregar o arquivo COMUNICAÇÃO: {e}")

    lista_de_dfs.append(df)
    df = pd.concat(lista_de_dfs, ignore_index=True)

lista_de_dfs_env = []
for uploaded_file2 in uploaded_files2:
    if uploaded_file2:
        try:
            df_env = pd.read_csv(uploaded_file2, encoding="iso-8859-1", delimiter=";", dtype={'Indexador': str})
            
            # Validação das colunas do arquivo ENTIDADES
            colunas_faltando_entidades = [col for col in colunas_esperadas_entidades if col not in df_env.columns]
            
            if colunas_faltando_entidades:
                st.error(
                    f"🚨 **Erro no arquivo ENTIDADES!** As seguintes colunas estão faltando: "
                    f"{', '.join(colunas_faltando_entidades)}. "
                    "Por favor, verifique se o arquivo está no formato correto e tente novamente."
                )
                df_env = None # Define como None para evitar processamento com dados incompletos
            else:
                st.success("✅ Arquivo ENTIDADES carregado e validado com sucesso!")
            match = re.search(r'(\d+)', uploaded_file2.name)
            if match:
                rif_num = match.group(1)
                df_env['rif_num'] = rif_num
            else:
                # Se não encontrar número, usa o nome do arquivo como fallback
                df_env['rif_num'] = uploaded_file.name

        except Exception as e:
            st.error(f"Ocorreu um erro ao carregar o arquivo ENTIDADES: {e}")

    lista_de_dfs_env.append(df_env)
    df_env = pd.concat(lista_de_dfs_env, ignore_index=True)

# Somente prossiga se ambos os DataFrames foram carregados e validados com sucesso
if df is not None and df_env is not None:
    st.write("---")

if uploaded_files and uploaded_files2:
    df = df[df.CodigoSegmento == "41"].reset_index(drop=True)
    coluna_escolhida = 'informacoesAdicionais'

    st.markdown("#### Seleção das comunicações")
    sel_comunicacoes = st.radio(
    "Verifica todas as comunicações ou apenas as selecionadas",
    ["Todas", "Seleção (Manual)", "Seleção (Automática)"],
    index=None, help="A seleção automática utiliza as comunicações que os investigados figuram como titular", horizontal=True)
    
    if coluna_escolhida in df.columns:
        lista_comunicacoes = df.idComunicacao.values.tolist()
        
        indexador = df.Indexador.values.tolist()
        rif_num = df.rif_num.values.tolist()
        # para criar lista envolvidos CPF/CNPJ e NOMES
        todos_envolvidos_cpf_cnpj = df_env.cpfCnpjEnvolvido.dropna().astype(str).tolist()
        todos_envolvidos_nomes = df_env.nomeEnvolvido.dropna().tolist()
        # lista única
        todos_envolvidos_cpf_cnpj = list(set(todos_envolvidos_cpf_cnpj))
        todos_envolvidos_nomes = list(set(todos_envolvidos_nomes))
        todos_envolvidos_nomes = [item.strip() for item in todos_envolvidos_nomes if len(item.strip()) >= 3]
        # dataframe com ni e nome apenas
        df_env_ni_nome = df_env[['cpfCnpjEnvolvido', 'nomeEnvolvido']].drop_duplicates().sort_values(by=['cpfCnpjEnvolvido', 'nomeEnvolvido']).reset_index(drop=True)



        lista_comunicacoes_editada = []
        for comunica in lista_comunicacoes:
            lista_comunicacoes_editada.append(str(comunica)[:-2])

        if sel_comunicacoes == "Todas":
            pass
        elif sel_comunicacoes == "Seleção (Manual)":
            options_comunic = st.multiselect(
            "Selecione todas as comunicações que deseja trabalhar (pelo menos duas)",
            lista_comunicacoes_editada)
            if options_comunic and len(options_comunic) > 1:
                lista_comunicacoes_editada = options_comunic
            else:
                st.error("Selecione mais de uma comunicação.")

        elif sel_comunicacoes == "Seleção (Automática)":
            options_alvos = st.multiselect(
            "Selecione todos os alvos da investigação",
            todos_envolvidos_cpf_cnpj)
            if options_alvos:
                df_env_sel_aut = df_env[df_env.cpfCnpjEnvolvido.isin(options_alvos)]
                df_env_sel_aut = df_env_sel_aut[df_env_sel_aut.tipoEnvolvido.str.strip().str.lower() == "titular"]
                df_env_sel_aut = df_env_sel_aut[["Indexador", "rif_num"]].drop_duplicates().reset_index(drop=True)
                if not df_env_sel_aut.empty:
                    merged_df_aut = pd.merge(df_env_sel_aut, df, on=['Indexador', 'rif_num'], how='inner')
                    lista_comunicacoes = merged_df_aut.idComunicacao.values.tolist()
                    lista_comunicacoes_editada = []
                    for comunica in lista_comunicacoes:
                        lista_comunicacoes_editada.append(str(comunica)[:-2])
                else:
                    st.error("Não há comunicações que atendem o critério. Mostrando todas.")

        st.markdown("##### Quantidade de comunicações na seleção: {}".format(len(lista_comunicacoes_editada)))
        if len(lista_comunicacoes_editada) > 1:
            comunic_numero = st.select_slider("Escolha a comunicação", options=lista_comunicacoes_editada)
            idx = df.loc[df['idComunicacao'] == float(comunic_numero)].index[0]
            ind = df.at[idx, "Indexador"]
            n_rif = df.at[idx, "rif_num"]
            texto_original = df.at[idx, coluna_escolhida]
            banco_comunicante = df.at[idx, "nomeComunicante"]
        elif len(lista_comunicacoes_editada) == 1:
            # CASO 2: Apenas uma comunicação. Não precisamos de slider.
            comunic_numero = lista_comunicacoes_editada[0] # Pega o único elemento
            st.info(f"Apenas uma comunicação encontrada: **{comunic_numero}**")
            
            idx = df.loc[df['idComunicacao'] == float(comunic_numero)].index[0]
            ind = df.at[idx, "Indexador"]
            n_rif = df.at[idx, "rif_num"]
            texto_original = df.at[idx, coluna_escolhida]
            banco_comunicante = df.at[idx, "nomeComunicante"]

        # para selecionar apenas os indexes da comunicação
        df_env = df_env[df_env['Indexador'].isin(indexador)]

        # st.dataframe(df_env) BRONCA AQUI (LEMBRAR DE ESPECIFICAR OS FORMATOS DAS COLUNAS NA IMPORTAÇÃO PARA EVITAR ERROS)

        st.dataframe(df_env[(df_env.Indexador == ind) & (df_env.rif_num == n_rif)])

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
                    try:                   
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
                    except Exception as e:
                        st.error(f"Ocorreu um erro ao tentar baixar o json: {e}")

        # Se ativado, exibe o expander com markdown renderizado
        if st.session_state["mostrar_dialog"]:
            expandir = st.expander("👁️ Visualização da comunicação em Markdown")
            with expandir:
                st.markdown(st.session_state.get("texto_transformado", ""), unsafe_allow_html=True)
        
        st.divider()
        if novo_texto:

            st.markdown("### 🧮 Tabela editável Origem/Destino dos recursos")

            # TESTE TABELA ------------------

            # Encontrar todos os blocos de tabela markdown
            try:
                tabela_origem = re.findall(r"((?:\|.*\n)+)", st.session_state["texto_transformado"].split('\n\n ------------ \n\n')[1])
                df_origem_final = juntar_tabelas_markdown(tabela_origem)
            except IndexError:
                df_origem_final = pd.DataFrame()
            try:
                tabela_destino = re.findall(r"((?:\|.*\n)+)", st.session_state["texto_transformado"].split('\n\n ------------ \n\n')[2])
                df_destino_final = juntar_tabelas_markdown(tabela_destino)
            except IndexError:
                df_destino_final = pd.DataFrame()
            # st.write(tabela_origem)

            
            
            # df_origem_final = pd.concat(dataframes, ignore_index=True)

            # st.write(df_origem_final)
            # st.write(df_destino_final)

            df_origem_destino_final = criar_tabela_origem_destino_preenchida(name_file, banco_comunicante, titular_env, df_origem_final, df_destino_final)

            # tratar dados presentes na tabela de maneira inicial
            # df_origem_destino_final = limpar_dataframe_ori_dest(df_origem_destino_final)
            # df_origem_destino_final["Remetente_CPF_CNPJ"] = df_origem_destino_final["Remetente_CPF_CNPJ"].apply(best_match, args=(df_env["cpfCnpjEnvolvido"].to_list(),))
            # df_origem_destino_final["Destinatário_CPF_CNPJ"] = df_origem_destino_final["Destinatário_CPF_CNPJ"].apply(best_match, args=(df_env["cpfCnpjEnvolvido"].to_list(),))
            
            
            st.write(df_origem_destino_final)
            st.download_button(
            "💾 Baixar tabela como CSV",
            df_origem_destino_final.to_csv(index=False, sep=';'),
            file_name="{}_origem_destino.csv".format(name_file),
            key="download_origem_destino")

            st.write("---")

        else:
            st.write("---")

        # FINAL TESTE ----------------

        # Adicionar opção para limpar e juntar origem e destino para a próxima opção de análise
        
        
        st.markdown("### Junte os arquivos origem_destino já baixados")
        uploaded_files_ori_dest = st.file_uploader("Faça upload de arquivos origem e destino", accept_multiple_files=True, type="csv")
        if uploaded_files_ori_dest:
            dfs_ori_dest = juntar_csvs(uploaded_files_ori_dest)
        
            df_limpo = limpar_dataframe_ori_dest(dfs_ori_dest)

            df_limpo["len_Remetente_CPF_CNPJ"] = df_limpo['Remetente_CPF_CNPJ'].apply(len_CPF_CNPJ)
            df_limpo["len_Destinatário_CPF_CNPJ"] = df_limpo['Destinatário_CPF_CNPJ'].apply(len_CPF_CNPJ)

            df_limpo['Remetente_CPF_CNPJ_sem_mascara'] = df_limpo['Remetente_CPF_CNPJ'].apply(limpar_documento)
            df_limpo['Destinatario_CPF_CNPJ_sem_mascara'] = df_limpo['Destinatário_CPF_CNPJ'].apply(limpar_documento)
            df_env_ni_nome['cpfCnpjEnvolvido_sem_mascara'] = df_env_ni_nome['cpfCnpjEnvolvido'].apply(limpar_documento)

            referencias = df_env_ni_nome['cpfCnpjEnvolvido_sem_mascara'].tolist()
            df_limpo['Remetente_CPF_CNPJ_sem_mascara_corrigido'] = df_limpo['Remetente_CPF_CNPJ_sem_mascara'].apply(lambda d: corrigir_doc(d, referencias))
            df_limpo['Destinatario_CPF_CNPJ_sem_mascara_corrigido'] = df_limpo['Destinatario_CPF_CNPJ_sem_mascara'].apply(lambda d: corrigir_doc(d, referencias))

            df_limpo['Destinatário_CPF_CNPJ'] = df_limpo['Destinatario_CPF_CNPJ_sem_mascara_corrigido'].apply(lambda x: aplicar_mascara(x))
            df_limpo['Remetente_CPF_CNPJ'] = df_limpo['Remetente_CPF_CNPJ_sem_mascara_corrigido'].apply(lambda x: aplicar_mascara(x))

            df_limpo_tratado = df_limpo.drop(['len_Remetente_CPF_CNPJ', 'len_Destinatário_CPF_CNPJ', 'Remetente_CPF_CNPJ_sem_mascara', 'Remetente_CPF_CNPJ_sem_mascara_corrigido', 'Destinatario_CPF_CNPJ_sem_mascara', 'Destinatario_CPF_CNPJ_sem_mascara_corrigido'], axis=1)
            
            st.write(df_limpo_tratado)
            numeracao_rif_str = "_".join("{}".format(r_number) for r_number in list(set(rif_num)))
            st.download_button(
            "💾 Baixar tabela como CSV",
            df_limpo_tratado.to_csv(index=False, sep=';'),
            file_name="origem_destino_tratado_rif_{}.csv".format(numeracao_rif_str),
            key="download_origem_destino_tabela_unica")

        else:
            st.markdown("Ainda sem arquivos juntados")


        # dataframe com ni e nome
        # df_env_ni_nome
        

    #     # Colunas e configurações
    #     colunas = ['Comunicação', 'Remetente_Nome', 'Remetente_CPF_CNPJ', 'Remetente_Banco', 'Valor', 'Quantidade', 'Destinatário_Nome', 'Destinatário_CPF_CNPJ', 'Destinatário_Banco']
    #     valores_padrao = {
    #         "Comunicação": name_file,
    #         "Remetente_Nome": "",
    #         "Remetente_CPF_CNPJ": "",
    #         "Remetente_Banco": "",
    #         "Valor": "",
    #         "Quantidade": "",
    #         "Destinatário_Nome":"",
    #         "Destinatário_CPF_CNPJ":"",
    #         "Destinatário_Banco":""
    #     }
    #     if "tabela_editavel" not in st.session_state:
    #         st.session_state["tabela_editavel"] = pd.DataFrame(columns=colunas)

    #     tabela = st.session_state["tabela_editavel"].copy()
    #     tabela.insert(0, "Linha", range(1, len(tabela) + 1))  # adiciona coluna numérica para referência



    #     # Configurar tabela interativa
    #     gb = GridOptionsBuilder.from_dataframe(tabela)
    #     gb.configure_default_column(editable=True, resizable=True)
    #     gb.configure_selection('multiple', use_checkbox=True)

    #     gb.configure_column("Linha", header_name="Linha", editable=False, width=70, pinned='left', checkboxSelection=True)

    #     # Coluna: Comunicação
    #     gb.configure_column("Comunicação", header_name="Comunicação", editable=True)
    #     # Coluna: Remetente_Nome (obrigatória)
    #     gb.configure_column("Remetente_Nome", header_name="Nome Remetente", editable=True, cellEditor="agTextCellEditor")
    #     # Coluna: Remetente_CPF_CNPJ
    #     gb.configure_column("Remetente_CPF_CNPJ", header_name="NI Remetente", editable=True)
    #     # Coluna: Remetente_Banco
    #     gb.configure_column("Remetente_Banco", header_name="Banco Remetente", editable=True)
    #      # Coluna: Valor (obrigatória)
    #     gb.configure_column("Valor", header_name="Valor", editable=True, type=["numericColumn","numberColumnFilter"], cellEditor="agTextCellEditor")
    #      # Coluna: Quantidade
    #     gb.configure_column("Quantidade", header_name="Quantidade", editable=True, type=["numericColumn","numberColumnFilter"])
    #      # Coluna: Destinatário_Nome (obrigatória)
    #     gb.configure_column("Destinatário_Nome", header_name="Nome Destinatário", editable=True, cellEditor="agTextCellEditor")
    #      # Coluna: Destinatário_CPF_CNPJ
    #     gb.configure_column("Destinatário_CPF_CNPJ", header_name="NI Destinatário", editable=True)
    #      # Coluna: Destinatário_Banco
    #     gb.configure_column("Destinatário_Banco", header_name="Banco Destinatário", editable=True)


    #     grid = AgGrid(
    #         tabela,
    #         gridOptions=gb.build(),
    #         editable=True,
    #         fit_columns_on_grid_load=True,
    #         theme="alpine",  # pode trocar para "streamlit", "material", "balham"
    #         update_mode=GridUpdateMode.MODEL_CHANGED,
    #         allow_unsafe_jscode=True,
    #         enable_enterprise_modules=False,
    #         height=400
    #     )

    #     # Atualizar dados com edição
    #     df_editado = pd.DataFrame(grid["data"]).drop(columns=["Linha"], errors="ignore")
    #     st.session_state["tabela_editavel"] = df_editado

    #     col1_tabela, col2_tabela, col3_tabela = st.columns([5, 1, 1])

    #     with col1_tabela:
        
    #         if st.button("🤯 Preencher automaticamente"):
    #             nova_linha = df_origem_destino_final
    #             st.session_state["tabela_editavel"] = pd.concat(
    #                 [st.session_state["tabela_editavel"], nova_linha], ignore_index=True
    #             )
    #             st.rerun()

    #     with col2_tabela:
    #         # Adicionar nova linha
    #         if st.button("➕ Adicionar linhas"):
                
    #             nova_linha = pd.DataFrame([valores_padrao])
    #             st.session_state["tabela_editavel"] = pd.concat(
    #                 [st.session_state["tabela_editavel"], nova_linha], ignore_index=True
    #             )
    #             st.rerun()

    #     # Não tá funcionando ------------- Criar função para apagar toda tabela tb
    #     with col3_tabela:
    #         if st.button("➖ Apagar linhas"):
    #             selecionadas = grid.get("selected_rows", [])
    #             if isinstance(selecionadas, list) and len(selecionadas) > 0:
    #                 # Obter índices das linhas selecionadas
    #                 linhas_para_remover = [row["Linha"] - 1 for row in selecionadas if "Linha" in row]
    #                 nova_df = st.session_state["tabela_editavel"].drop(index=linhas_para_remover).reset_index(drop=True)
    #                 st.session_state["tabela_editavel"] = nova_df
    #                 st.rerun()


    




    #     # st.session_state["tabela_editavel"] = df_editado

    #     # Validação simples antes de salvar
    #     erros = []

    #     for i, row in df_editado.iterrows():
    #         if not row["Remetente_Nome"] or not row["Valor"] or not row["Destinatário_Nome"]:
    #             erros.append(f"Linha {i+1} incompleta (Remetente_Nome, Valor e Destinatário_Nome são obrigatórios)")

    #     if erros:
    #         for e in erros:
    #             st.error(e)
    #     else:
    #         # Baixar CSV
    #         st.download_button(
    #             "💾 Baixar tabela como CSV",
    #             pd.DataFrame(df_editado).to_csv(index=False),
    #             file_name="{}_origem_destino.csv".format(name_file)
    #     )
    # else:
    #     st.warning("⚠️ Nenhuma coluna de texto encontrada no CSV.")
