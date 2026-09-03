
 """
Previsão de Situação Escolar
----------------------------
Treina um modelo de classificação (Decision Tree) para prever se um aluno
será Aprovado, Reprovado ou fica em Recuperação.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

import gradio as gr

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# 1. Dados
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

    # Ruído: embaralha ~5% dos rótulos
    n_ruido = max(1, int(0.05 * n_amostras))
    idx_ruido = rng.choice(df.index, size=n_ruido, replace=False)
    opcoes = ["Aprovado", "Reprovado", "Recuperação"]
    df.loc[idx_ruido, "Situacao"] = rng.choice(opcoes, size=n_ruido)

    return df


# --------------------------------------------------------------------------
# 2. Treinamento e avaliação
# --------------------------------------------------------------------------
def treinar_modelo(df: pd.DataFrame):
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
    modelo.fit(x_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores_cv = cross_val_score(modelo, x, y, cv=cv, scoring="accuracy")

    y_pred = modelo.predict(x_teste)

    print("=" * 60)
    print(f"Acurácia média (5-fold CV): {scores_cv.mean():.2%} (+/- {scores_cv.std():.2%})")
    print("-" * 60)
    print("Relatório de classificação (conjunto de teste):")
    print(classification_report(y_teste, y_pred))
    print("=" * 60)

    return modelo, x, y, x_teste, y_teste, y_pred, scores_cv


def salvar_matriz_confusao(y_teste, y_pred, classes, caminho="matriz_confusao.png"):
    cm = confusion_matrix(y_teste, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def salvar_arvore(modelo, feature_names, class_names, caminho="arvore_decisao.png"):
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
    plt.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


# --------------------------------------------------------------------------
# 3. Interface Gradio
# --------------------------------------------------------------------------
CORES = {
    "Aprovado": "#16a34a",
    "Recuperação": "#d97706",
    "Reprovado": "#dc2626",
}


def construir_interface(modelo, classes, path_cm, path_tree):

    def prever_situacao(horas, faltas, nota):
        horas = float(np.clip(horas, 0, 168))
        faltas = int(np.clip(faltas, 0, 365))
        nota = float(np.clip(nota, 0, 10))

        df_novo = pd.DataFrame(
            [[horas, faltas, nota]],
            columns=["Horas_de_estudo", "Faltas", "Nota"],
        )
        proba = modelo.predict_proba(df_novo)[0]
        pred = modelo.classes_[int(np.argmax(proba))]

        cor = CORES.get(pred, "#334155")
        resultado_html = f"""
        <div style="padding:18px 22px;border-radius:12px;background:{cor}1a;
                    border:1px solid {cor};text-align:center;">
            <div style="font-size:14px;color:#475569;margin-bottom:4px;">
                Previsão do modelo
            </div>
            <div style="font-size:26px;font-weight:700;color:{cor};">
                {pred}
            </div>
        </div>
        """

        prob_df = pd.DataFrame({
            "Situação": modelo.classes_,
            "Probabilidade": proba,
        }).sort_values("Probabilidade", ascending=False)

        return resultado_html, prob_df

    with gr.Blocks(
        title="Previsão de Situação Escolar",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    ) as interface:

        gr.Markdown(
            """
            # 🎓 Previsor de Situação Escolar
            Informe os dados do aluno para estimar se ele será **Aprovado**,
            ficará em **Recuperação** ou será **Reprovado**.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                horas = gr.Slider(0, 20, value=8, step=0.5, label="Horas de estudo por semana")
                faltas = gr.Slider(0, 30, value=3, step=1, label="Número de faltas")
                nota = gr.Slider(0, 10, value=7, step=0.1, label="Nota")
                botao = gr.Button("Prever situação", variant="primary")

                gr.Examples(
                    examples=[
                        [10, 2, 8.5],
                        [2, 15, 3.0],
                        [5, 6, 6.5],
                        [8, 1, 9.0],
                        [1, 20, 2.5],
                    ],
                    inputs=[horas, faltas, nota],
                    label="Exemplos rápidos",
                )

            with gr.Column(scale=1):
                resultado = gr.HTML(label="Resultado")
                grafico_proba = gr.BarPlot(
                    x="Situação",
                    y="Probabilidade",
                    title="Confiança do modelo por classe",
                    y_lim=[0, 1],
                    height=280,
                )

        # Atualizações dinâmicas
        entradas = [horas, faltas, nota]
        saidas = [resultado, grafico_proba]
        
        botao.click(fn=prever_situacao, inputs=entradas, outputs=saidas)
        for entrada in entradas:
            entrada.change(fn=prever_situacao, inputs=entradas, outputs=saidas)

        # Carrega visualização inicial na abertura da página
        interface.load(fn=prever_situacao, inputs=entradas, outputs=saidas)

        with gr.Accordion("📊 Artefatos e Métricas do Modelo", open=False):
            gr.Markdown(f"Classes previstas pelo modelo: **{', '.join(classes)}**")
            with gr.Row():
                gr.Image(value=path_cm, label="Matriz de Confusão", show_label=True)
                gr.Image(value=path_tree, label="Árvore de Decisão Gerada", show_label=True)

    return interface


# --------------------------------------------------------------------------
# 4. Execução principal
# --------------------------------------------------------------------------
def main():
    df = gerar_dataset_sintetico(n_amostras=300)
    modelo, x, y, x_teste, y_teste, y_pred, scores_cv = treinar_modelo(df)

    path_cm = salvar_matriz_confusao(y_teste, y_pred, classes=modelo.classes_)
    path_tree = salvar_arvore(modelo, feature_names=list(x.columns), class_names=list(modelo.classes_))

    interface = construir_interface(modelo, classes=list(modelo.classes_), path_cm=path_cm, path_tree=path_tree)
    interface.launch()


if __name__ == "__main__":
    main()
