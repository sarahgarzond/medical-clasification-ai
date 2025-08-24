"""
Demo interactivo del clasificador de literatura médica (multi-label)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, json, io
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go

# Para métricas (si el usuario quiere calcular)
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, classification_report

# -------------------------
# Configuración de la página
# -------------------------
st.set_page_config(page_title="Clasificador de Literatura Médica", page_icon="🏥", layout="wide")
st.title("🏥 Clasificador de Literatura Médica (Multi-label)")
st.markdown("Demo y análisis — utiliza tu modelo local HuggingFace (multi-label).")

# -------------------------
# Cargar modelo (desde carpeta local)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_classifier(MODEL_PATH = "C:/Users/user/Downloads/medical-literature-classifier/pubmedbert_model_final_2/pubmedbert_model_final"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(MODEL_PATH):
        return None, None, None, f"Ruta de modelo no encontrada: {MODEL_PATH}"

    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        return None, None, None, f"Error cargando modelo/tokenizer: {e}"

    model.to(device)
    # thresholds: archivo best_thresholds.json con {"classes": [...], "thresholds": [...]}
    thresholds_path = os.path.join(MODEL_PATH, "best_thresholds.json")
    if not os.path.exists(thresholds_path):
        # fallback: si no hay thresholds, usamos 0.5 por defecto y tratamos que CLASSES venga del config o del model config
        CLASSES = getattr(model.config, "id2label", None)
        if CLASSES is None:
            return model, tokenizer, None, "No se encontró best_thresholds.json ni id2label en la config del modelo."
        # id2label es dict map idx->label
        # convertir a lista ordenada por idx
        CLASSES = [CLASSES[str(i)] if str(i) in CLASSES else CLASSES[i] for i in range(len(CLASSES))]
        THRESHOLDS = [0.5] * len(CLASSES)
    else:
        with open(thresholds_path, "r", encoding="utf-8") as f:
            th_data = json.load(f)
        CLASSES = th_data.get("classes", None)
        THRESHOLDS = th_data.get("thresholds", None)
        if CLASSES is None or THRESHOLDS is None:
            return model, tokenizer, None, "best_thresholds.json no contiene 'classes' o 'thresholds'."

    def predict(texts, max_len=512, batch_size=16):
        """
        texts: list[str]
        devuelve: list de listas de dicts: [{"class": cls, "probability": float, "predicted": 0/1}, ...]
        """
        device = next(model.parameters()).device
        all_probs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(device)
                outputs = model(**enc)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()  # shape (batch_size, n_classes)
                all_probs.append(probs)
        all_probs = np.vstack(all_probs)
        results = []
        for row in all_probs:
            row_res = []
            for cls, prob, th in zip(CLASSES, row, THRESHOLDS):
                row_res.append({"class": cls, "probability": float(prob), "predicted": int(float(prob) > float(th))})
            results.append(row_res)
        return results

    return model, tokenizer, (CLASSES, THRESHOLDS, predict), None

# Cambia aquí la ruta si tu carpeta tiene otro nombre o ubicación
MODEL_PATH = "./pubmedbert_model_final"
model, tokenizer, predict_pack, load_error = load_classifier(MODEL_PATH)
if predict_pack is not None:
    CLASSES, THRESHOLDS, classifier = predict_pack
else:
    CLASSES, THRESHOLDS, classifier = None, None, None

# -------------------------
# Sidebar: navegación y opciones globales
# -------------------------
st.sidebar.title("Navegación")
page = st.sidebar.selectbox("Selecciona una página", ["Demo de Clasificación", "Análisis de Resultados", "Métricas del Modelo"])
st.sidebar.markdown("---")
st.sidebar.write("Ruta del modelo:")
st.sidebar.code(MODEL_PATH)
if load_error:
    st.sidebar.error(load_error)

# -------------------------
# Helper utilities
# -------------------------
def preds_list_to_df(preds_list):
    """Convierte output de classifier (lista de filas) a DataFrame con columnas prob_<cls>, pred_<cls>."""
    rows = []
    for row in preds_list:
        probs = {f"prob_{d['class']}": d['probability'] for d in row}
        preds = {f"pred_{d['class']}": d['predicted'] for d in row}
        rows.append({**probs, **preds})
    return pd.DataFrame(rows)

def top_pred_info(row_preds):
    """Dado row_preds (lista de dicts) devuelve top_class, top_prob, top_pred_flag"""
    sorted_by_prob = sorted(row_preds, key=lambda x: x["probability"], reverse=True)
    top = sorted_by_prob[0]
    return top["class"], top["probability"], top["predicted"]

# -------------------------
# Página: Demo de Clasificación
# -------------------------
if page == "Demo de Clasificación":
    st.header("🔍 Demo de Clasificación")

    if model is None or classifier is None:
        st.error("No se pudo cargar el modelo. Revisa el panel izquierdo con el error.")
        st.stop()

    st.markdown("Puedes ingresar un artículo (título + abstract), usar ejemplos de muestra, o cargar un CSV con columnas `title` y `abstract`.")

    tab1, tab2 = st.tabs(["Clasificar un artículo", "Cargar CSV / Batch"])

    # --------------
    # Tab: Single article
    # --------------
    with tab1:
        st.subheader("Clasificar un artículo (manual / sample)")
        c1, c2 = st.columns([2,1])
        with c1:
            title = st.text_input("Título del artículo:")
            abstract = st.text_area("Abstract:", height=180)
        with c2:
            st.markdown("### Opciones")
            use_sample = st.checkbox("Usar artículo de muestra (`Try sample articles`)", value=False)
            sample_selector = None
            sample_df = None
            if use_sample:
                # Intentamos cargar samples prehechos si existen en data/samples.csv
                sample_path = "data/sample_articles.csv"
                if os.path.exists(sample_path):
                    sample_df = pd.read_csv(sample_path)
                    # esperar que sample_df tenga columnas title, abstract
                    if 'title' in sample_df.columns and 'abstract' in sample_df.columns:
                        sample_titles = (sample_df['title'].astype(str) + " — " + sample_df.get('abstract', '').astype(str)).tolist()
                        sample_index = st.selectbox("Selecciona un sample", options=list(range(len(sample_titles))), format_func=lambda i: sample_titles[i][:120])
                        sample_selector = sample_index
                    else:
                        st.info("sample_articles.csv no tiene 'title' y 'abstract'.")
                else:
                    # si no existe file, mostramos 3 ejemplos de dummy
                    dummy = [
                        ("Neurological case study about seizure and EEG", "We report a patient with recurrent seizures..."),
                        ("Hepatorenal syndrome: a new therapy", "Study shows improvement in kidney function after..."),
                        ("Novel oncological therapy for melanoma", "Phase II trial demonstrates tumor shrinkage...")
                    ]
                    idx = st.selectbox("Selecciona un sample (ejemplo)", range(len(dummy)), format_func=lambda i: dummy[i][0])
                    sample_selector = idx
                    sample_df = pd.DataFrame(dummy, columns=['title','abstract'])

            if st.button("Cargar sample al formulario") and use_sample and sample_selector is not None:
                title = str(sample_df.loc[sample_selector, 'title'])
                abstract = str(sample_df.loc[sample_selector, 'abstract'])
                st.experimental_rerun()  # para reflejarlo en el formulario (simple)

            st.markdown("---")
            max_len = st.number_input("Max token length (truncation)", min_value=64, max_value=2048, value=512, step=64)
            st.write("Umbrales por clase (desde best_thresholds.json):")
            # Mostrar umbrales
            th_table = pd.DataFrame({"class": CLASSES, "threshold": THRESHOLDS})
            st.dataframe(th_table, height=200)

        # Botón clasificar article
        if st.button("Clasificar Artículo"):
            if not title or not abstract:
                st.warning("Por favor, ingresa título y abstract (o usa un sample).")
            else:
                with st.spinner("Obteniendo predicciones..."):
                    text = title + " " + abstract
                    preds = classifier([text], max_len=max_len)[0]
                    # DataFrame de resultados
                    res_df = pd.DataFrame(preds)
                    # Mostrar tabla
                    st.subheader("Resultados (por clase)")
                    st.dataframe(res_df.style.format({"probability": "{:.3f}"}))

                    # Barras de probabilidad
                    fig = px.bar(res_df, x="class", y="probability", color="predicted",
                                 title="Probabilidades por clase (barra) - las etiquetas con predicted=1 superan su umbral")
                    st.plotly_chart(fig, use_container_width=True)

                    # Indicador tipo gauge: mostrar la probabilidad máxima (útil como 'confianza')
                    top_class, top_prob, top_pred_flag = top_pred_info(preds)
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=top_prob*100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"Confianza (top) — {top_class}"},
                        delta={'reference': 50, 'relative': False},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                        }
                    ))
                    st.plotly_chart(fig_g, use_container_width=True)

                    # Mostrar etiquetas activas
                    active_labels = [d['class'] for d in preds if d['predicted'] == 1]
                    st.markdown(f"**Etiquetas activas (por encima del umbral):** {', '.join(active_labels) if active_labels else 'Ninguna'}")

    # --------------
    # Tab: CSV / Batch
    # --------------
    with tab2:
        st.subheader("Clasificar múltiples artículos (CSV)")
        st.markdown("Carga un CSV con columnas `title` y `abstract`.\nSi además tienes etiquetas verdaderas para evaluar, añade columnas `true_<class>` con 0/1 o una columna `labels` con lista/coma-sep.")
        uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Archivo cargado: {len(df)} registros")
                st.dataframe(df.head())
                # Validar que tenga title & abstract
                if 'title' not in df.columns or 'abstract' not in df.columns:
                    st.warning("El CSV debe tener columnas 'title' y 'abstract'.")
                else:
                    if st.button("Clasificar Todos los Artículos"):
                        texts = (df['title'].astype(str) + " " + df['abstract'].astype(str)).tolist()
                        with st.spinner("Clasificando..."):
                            preds = classifier(texts)
                            preds_df = preds_list_to_df(preds)
                            # join
                            out_df = pd.concat([df.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)
                        st.success("¡Clasificación completada!")
                        st.dataframe(out_df.head())

                        # Descarga
                        csv = out_df.to_csv(index=False)
                        st.download_button("Descargar resultados CSV", data=csv, file_name="clasificacion_multilabel_results.csv", mime="text/csv")

                        # Visualizaciones globales
                        # Conteo por clase (número de artículos con pred=1)
                        counts = {cls: int(out_df[f"pred_{cls}"].sum()) for cls in CLASSES}
                        fig_counts = px.bar(x=list(counts.keys()), y=list(counts.values()),
                                           title="Número de artículos predichos por clase")
                        st.plotly_chart(fig_counts, use_container_width=True)

                        # Histograma de probabilidades (todas las clases aplanadas)
                        probs_flat = []
                        for cls in CLASSES:
                            probs_flat.extend(out_df[f"prob_{cls}"].tolist())
                        fig_hist = px.histogram(x=probs_flat, nbins=40, title="Distribución de probabilidades (todas las clases)")
                        st.plotly_chart(fig_hist, use_container_width=True)

                        # Si el CSV incluye etiquetas verdaderas, calcular métricas multi-label
                        # Soportamos: columnas true_<class> (0/1) o columna 'labels' con coma-sep.
                        true_cols = [c for c in out_df.columns if c.startswith("true_")]
                        if true_cols:
                            st.info("Se han detectado columnas de verdad (true_<class>) — calculando métricas multi-label.")
                            # construir y_true y y_pred matrices
                            y_true = out_df[true_cols].values.astype(int)
                            y_pred = out_df[[f"pred_{c[5:]}" for c in true_cols]].values.astype(int)
                            # metrics por clase
                            p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
                            metrics_df = pd.DataFrame({
                                "class": [c[5:] for c in true_cols],
                                "precision": p,
                                "recall": r,
                                "f1": f1,
                                "support": sup
                            })
                            st.subheader("Métricas por clase (desde CSV)")
                            st.dataframe(metrics_df)
                        else:
                            st.info("No se encontraron columnas 'true_<class>'. Si quieres evaluar, añade esas columnas con 0/1 o una columna 'labels' con etiquetas.")
            except Exception as e:
                st.error(f"Error procesando el CSV: {e}")

# -------------------------
# Página: Análisis de Resultados
# -------------------------
elif page == "Análisis de Resultados":
    st.header("📊 Análisis de Resultados")
    st.markdown("Aquí se muestran resultados precomputados si existen (por ejemplo `results/results.json`) o puedes subir un CSV con verdaderos vs predichos para generar reportes.")

    results_path = "results/results.json"
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            st.subheader("Métricas principales (guardadas)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{results.get('accuracy', 0):.3f}")
            with col2:
                st.metric("F1-Score (Weighted)", f"{results.get('f1_weighted', 0):.3f}")
            with col3:
                st.metric("F1-Score (Macro)", f"{results.get('f1_macro', 0):.3f}")
            with col4:
                st.metric("F1-Score (Micro)", f"{results.get('f1_micro', 0):.3f}")

            st.subheader("Matriz de confusión (si existe)")
            if 'confusion_matrix' in results and 'class_names' in results:
                cm = np.array(results['confusion_matrix'])
                class_names = results['class_names']
                fig_cm = px.imshow(cm, labels=dict(x="Predicción", y="Valor Real", color="Cantidad"),
                                   x=class_names, y=class_names, color_continuous_scale='Blues', text_auto=True)
                fig_cm.update_layout(title="Matriz de Confusión")
                st.plotly_chart(fig_cm, use_container_width=True)

            st.subheader("Classification Report (guardado)")
            if 'classification_report' in results:
                report = results['classification_report']
                # convertir a df cuando es dict de dicts
                keys = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
                if keys:
                    metrics_df = pd.DataFrame({
                        'Clase': keys,
                        'Precision': [report[k]['precision'] for k in keys],
                        'Recall': [report[k]['recall'] for k in keys],
                        'F1-Score': [report[k]['f1-score'] for k in keys],
                        'Support': [report[k]['support'] for k in keys]
                    })
                    st.dataframe(metrics_df)
        except Exception as e:
            st.error(f"Error leyendo results.json: {e}")
    else:
        st.info("No se encontró `results/results.json`. Puedes subir un CSV con 'title','abstract' y 'true_<class>' para generar un análisis.")
        upload_eval = st.file_uploader("Sube un CSV para análisis (opcional)", type=['csv'])
        if upload_eval is not None:
            try:
                eval_df = pd.read_csv(upload_eval)
                st.write(f"Registros: {len(eval_df)}")
                if 'title' not in eval_df.columns or 'abstract' not in eval_df.columns:
                    st.warning("El CSV debe tener 'title' y 'abstract' además de columnas 'true_<class>' opcionales.")
                else:
                    texts = (eval_df['title'].astype(str) + " " + eval_df['abstract'].astype(str)).tolist()
                    with st.spinner("Clasificando para análisis..."):
                        preds = classifier(texts)
                        preds_df = preds_list_to_df(preds)
                        out_df = pd.concat([eval_df.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)
                    st.success("Clasificación para análisis completada.")
                    st.dataframe(out_df.head())

                    # Si encuentra true_<class> calcula métricas multi-label
                    true_cols = [c for c in out_df.columns if c.startswith("true_")]
                    if true_cols:
                        y_true = out_df[true_cols].values.astype(int)
                        y_pred = out_df[[f"pred_{c[5:]}" for c in true_cols]].values.astype(int)
                        p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
                        metrics_df = pd.DataFrame({
                            "class": [c[5:] for c in true_cols],
                            "precision": p, "recall": r, "f1": f1, "support": sup
                        })
                        st.subheader("Métricas por clase (desde CSV)")
                        st.dataframe(metrics_df)
                        st.subheader("Matriz de confusión (por clase combinada - no completamente informativa para multi-label)")
                        # matriz de confusión agregada: convertir multi-hot a etiqueta de mayor prob para propósito ilustrativo
                        # NOTA: Para multi-label la matriz de confusión tradicional no es exactamente aplicable.
                        y_true_single = np.argmax(y_true, axis=1)
                        y_pred_single = np.argmax(y_pred, axis=1)
                        cm = confusion_matrix(y_true_single, y_pred_single)
                        fig_cm = px.imshow(cm, labels=dict(x="Predicción (argmax)", y="Real (argmax)", color="Cantidad"),
                                           x=[c[5:] for c in true_cols], y=[c[5:] for c in true_cols], text_auto=True)
                        st.plotly_chart(fig_cm, use_container_width=True)
                    else:
                        st.info("No se encontraron columnas 'true_<class>' en el CSV.")
            except Exception as e:
                st.error(f"Error procesando CSV de análisis: {e}")

# -------------------------
# Página: Métricas del Modelo
# -------------------------
elif page == "Métricas del Modelo":
    st.header("🎯 Métricas del Modelo y Características")
    st.markdown("Aquí puedes ver información sobre el entrenamiento, características (si las guardaste) y métricas.")

    # Feature importance (si existe archivo)
    feat_path = 'results/feature_importance.csv'
    if os.path.exists(feat_path):
        try:
            ft = pd.read_csv(feat_path)
            st.subheader("Características Más Importantes")
            st.dataframe(ft.head(50))
            fig_feat = px.bar(ft.head(20), x='importance', y='feature', orientation='h', title="Top 20 Características")
            fig_feat.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_feat, use_container_width=True)
        except Exception as e:
            st.error(f"Error leyendo feature_importance.csv: {e}")
    else:
        st.info("No se encontró 'results/feature_importance.csv'.")

    # Información del modelo
    st.subheader("Información del Modelo (cargado)")
    if model is not None:
        st.write("**Tipo de modelo:** HuggingFace Transformer (AutoModelForSequenceClassification)")
        st.write(f"**Número de clases (salida):** {len(CLASSES) if CLASSES is not None else 'Desconocido'}")
        st.write("**Modelo cargado desde:**")
        st.code(MODEL_PATH)
    else:
        st.warning("Modelo no cargado.")

    # Mostrar umbrales y clases
    if CLASSES is not None:
        st.subheader("Clases y Umbrales")
        th_table = pd.DataFrame({"class": CLASSES, "threshold": THRESHOLDS})
        st.dataframe(th_table)

    # Distribución de clases en un train.csv si existe
    train_path = 'data/train.csv'
    if os.path.exists(train_path):
        try:
            train_df = pd.read_csv(train_path)
            if 'group' in train_df.columns:
                class_counts = train_df['group'].value_counts()
                fig = px.bar(x=class_counts.index, y=class_counts.values, labels={'x': 'Grupo', 'y': 'Cantidad'}, title="Distribución de clases (train.csv)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("train.csv no contiene columna 'group'.")
        except Exception as e:
            st.error(f"Error leyendo train.csv: {e}")
    else:
        st.info("No se encontró data/train.csv para mostrar distribución.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("🏥 **Clasificador de Literatura Médica** — Multi-label. Asegúrate de poner tu carpeta del modelo en `MODEL_PATH` y que incluya `best_thresholds.json` con `classes` y `thresholds`.")