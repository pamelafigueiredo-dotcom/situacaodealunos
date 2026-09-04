"""
Previsão de Situação Escolar
----------------------------
Treina um modelo de classificação (Decision Tree) para prever se um aluno
será Aprovado, Reprovado ou fica em Recuperação.
Interface construída com Streamlit.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

import streamlit as st

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Configuração da página Streamlit
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Previsão de Situação Escolar",
    page_icon="🎓",
    layout="wide"
)

CORES = {
    "Aprovado": "#16a34a",
    "Recuperação": "#d97706",
    "Reprovado": "#dc2626",
}


# --------------------------------------------------------------------------
# 1. Dados Sintéticos
# --------------------------------------------------------------------------
def gerar_dataset_sintetico(n_amostras: int = 300, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    horas_estudo = rng.uniform(0, 15, n_amostras).round(1)
    faltas = rng.integers(0, 25, n_amostras)
    nota = rng.uniform(0, 10, n_amostras).round(1)

    situacao = []
    for h, f, n in zip(horas_estudo, faltas, nota):
        if f > 18:
            situacao.append("Reprovado")
        elif n >= 7 and f <= 10:
            situacao.append("Aprovado")
        elif n < 4:
            situacao.append("Reprovado")
        elif 4 <= n < 7:
            situacao.append("Recuperação")
        else:
            situacao.append("Aprovado")

    df = pd.DataFrame({
        "Horas_de_estudo": horas_estudo,
        "Faltas": faltas,
        "Nota": nota,
        "Situacao": situacao,
    })

    # Adiciona ruído alterando o rótulo original
    n_ruido = max(1, int(0.05 * n_amostras))
    idx_ruido = rng.choice(df.index, size=n_ruido, replace=False)
    opcoes = np.array(["Aprovado", "Reprovado", "Recuperação"])

    for idx in idx_ruido:
        atual = df.at[idx, "Situacao"]
        novas_opcoes = opcoes[opcoes != atual]
        df.at[idx, "Situacao"] = rng.choice(novas_opcoes)

    return df


# --------------------------------------------------------------------------
# 2. Treinamento do Modelo (com Cache)
# --------------------------------------------------------------------------
@st.cache_resource
def carregar_e_treinar_modelo():
    df = gerar_dataset_sintetico(n_amostras=300)
    x = df[["Horas_de_estudo", "Faltas", "Nota"]]
    y = df["Situacao"]

    x_train, x_teste, y_train, y_teste = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    modelo = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores_cv = cross_val_score(modelo, x_train, y_train, cv=cv, scoring="accuracy")

    modelo.fit(x_train, y_train)
    y_pred = modelo.predict(x_teste)

    report_dict = classification_report(y_teste, y_pred, output_dict=True)

    return modelo, x, y_teste, y_pred, scores_cv, report_dict


def gerar_figura_matriz_confusao(y_teste, y_pred, classes):
    cm = confusion_matrix(y_teste, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    return fig


def gerar_figura_arvore(modelo, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        modelo,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 3. Interface Streamlit
# --------------------------------------------------------------------------
def main():
    st.title("🎓 Previsor de Situação Escolar")
    st.markdown(
        "Informe os dados do aluno para estimar se ele será **Aprovado**, "
        "ficará em **Recuperação** ou será **Reprovado**."
    )

    # Carrega dados e treina o modelo
    modelo, x, y_teste, y_pred, scores_cv, report_dict = carregar_e_treinar_modelo()

    # --- Barra Lateral (Inputs) ---
    st.sidebar.header("📋 Dados do Aluno")

    # Exemplos pré-definidos
    exemplo = st.sidebar.selectbox(
        "Carregar Exemplo Rápido",
        [
            "Personalizado",
            "Aluno Aprovado (10h, 2 faltas, Nota 8.5)",
            "Aluno Reprovado por falta (2h, 15 faltas, Nota 3.0)",
            "Aluno Recuperação (5h, 6 faltas, Nota 6.5)",
            "Aluno Excelente (8h, 1 falta, Nota 9.0)",
            "Aluno Faltoso (1h, 20 faltas, Nota 2.5)",
        ]
    )

    # Valores padrão iniciais
    val_horas, val_faltas, val_nota = 8.0, 3, 7.0

    if exemplo == "Aluno Aprovado (10h, 2 faltas, Nota 8.5)":
        val_horas, val_faltas, val_nota = 10.0, 2, 8.5
    elif exemplo == "Aluno Reprovado por falta (2h, 15 faltas, Nota 3.0)":
        val_horas, val_faltas, val_nota = 2.0, 15, 3.0
    elif exemplo == "Aluno Recuperação (5h, 6 faltas, Nota 6.5)":
        val_horas, val_faltas, val_nota = 5.0, 6, 6.5
    elif exemplo == "Aluno Excelente (8h, 1 falta, Nota 9.0)":
        val_horas, val_faltas, val_nota = 8.0, 1, 9.0
    elif exemplo == "Aluno Faltoso (1h, 20 faltas, Nota 2.5)":
        val_horas, val_faltas, val_nota = 1.0, 20, 2.5

    horas = st.sidebar.slider("Horas de estudo por semana", 0.0, 20.0, val_horas, step=0.5)
    faltas = st.sidebar.slider("Número de faltas", 0, 30, val_faltas, step=1)
    nota = st.sidebar.slider("Nota", 0.0, 10.0, val_nota, step=0.1)

    # --- Área Principal (Previsão) ---
    col1, col2 = st.columns([1, 1])

    df_novo = pd.DataFrame(
        [[horas, faltas, nota]],
        columns=["Horas_de_estudo", "Faltas", "Nota"],
    )
    proba = modelo.predict_proba(df_novo)[0]
    pred = modelo.classes_[int(np.argmax(proba))]
    cor = CORES.get(pred, "#334155")

    with col1:
        st.subheader("Resultado da Previsão")
        st.markdown(
            f"""
            <div style="padding:22px; border-radius:12px; background:{cor}1a;
                        border:2px solid {cor}; text-align:center;">
                <div style="font-size:14px; color:#475569; margin-bottom:4px;">
                    Previsão do modelo
                </div>
                <div style="font-size:32px; font-weight:700; color:{cor};">
                    {pred}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("Confiança do Modelo")
        prob_df = pd.DataFrame({
            "Situação": modelo.classes_,
            "Probabilidade": proba,
        }).set_index("Situação")

        st.bar_chart(prob_df, y="Probabilidade", height=200)

    # --- Seção Expansível (Artefatos e Métricas) ---
    st.markdown("---")
    with st.expander("📊 Artefatos e Métricas do Modelo"):
        st.write(f"**Acurácia média (5-fold CV):** {scores_cv.mean():.2%} (+/- {scores_cv.std():.2%})")
        st.write(f"Classes previstas pelo modelo: **{', '.join(modelo.classes_)}**")

        tab1, tab2, tab3 = st.tabs(["Matriz de Confusão", "Árvore de Decisão", "Relatório de Classificação"])

        with tab1:
            fig_cm = gerar_figura_matriz_confusao(y_teste, y_pred, modelo.classes_)
            st.pyplot(fig_cm)

        with tab2:
            fig_tree = gerar_figura_arvore(modelo, list(x.columns), list(modelo.classes_))
            st.pyplot(fig_tree)

        with tab3:
            st.dataframe(pd.DataFrame(report_dict).transpose())


if __name__ == "__main__":
    main()
