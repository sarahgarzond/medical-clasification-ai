"""
Script para crear datos de muestra basados en el dataset proporcionado
"""

import pandas as pd
import numpy as np

def create_sample_dataset():
    """
    Crea un dataset de muestra basado en la imagen proporcionada
    """
    
    # Datos de muestra basados en la imagen del Excel
    sample_data = {
        'title': [
            'Haim-Munk: Of the many neurological',
            'Exploring lur Background: hepatorenal',
            'High fat diet Often, chem cardiovascular hepatorenal',
            'Influence of Aniridia in m neurological',
            'The effect of The effectiv neurological',
            'When palsy i Research qu neurological',
            'Cardioprotec The present cardiovascular hepatorenal',
            'L-arginine tr A deficient L cardiovascular',
            'The hunting Hypothesis: neurological',
            'Study of the Benign famil neurological',
            'Expression o The von Hipp neurological hepatorenal oncological',
            'Tumor terrai Background: oncological',
            'gaba pathwa Background: neurological',
            'Metformin p The antidiab neurological hepatorenal',
            'Germline BR The objectiv neurological oncological',
            'The nephrol Hypothesis: neurological hepatorenal',
            'Brain mappi Hypothesis: neurological',
            'Complex gly Five male Ja neurological hepatorenal',
            'Genetic hete The Breast C oncological',
            'defibrillato Objective: T cardiovascular',
            'Hyperosmol A 45-year-ol hepatorenal',
            'Pioneering c Research qu hepatorenal',
            'When convu Background: neurological oncological',
            'Granulosa ce BACKGROUN hepatorenal oncological',
            'Behavioral a Garcinellipt neurological',
            'epilepsy anc Study design cardiovascular',
            'Anti-oxidant 1. Dexameth neurological cardiovascular',
            'cardiac glyc Research qu neurological'
        ],
        'abstract': [
            'This study examines neurological disorders and their impact on patient outcomes in clinical settings.',
            'Background research on hepatorenal syndrome and its treatment approaches in modern medicine.',
            'Investigation of high fat diet effects on cardiovascular and hepatorenal systems in patients.',
            'Analysis of Aniridia influence on neurological development and associated complications.',
            'Research on the effectiveness of neurological interventions in patient care and recovery.',
            'Study of palsy research questions and their neurological implications for treatment.',
            'Cardioprotective mechanisms and their role in cardiovascular and hepatorenal health.',
            'L-arginine deficiency effects on cardiovascular system and therapeutic interventions.',
            'Hunting hypothesis examination in neurological disorders and disease progression.',
            'Benign familial neurological conditions and their clinical manifestations in patients.',
            'Expression analysis of von Hippel genes in neurological, hepatorenal and oncological contexts.',
            'Tumor terrain background research in oncological studies and cancer progression.',
            'GABA pathway analysis in neurological disorders and neurotransmitter function.',
            'Metformin antidiabetic effects on neurological and hepatorenal systems.',
            'Germline mutations objective analysis in neurological and oncological research.',
            'Nephrology hypothesis testing in neurological and hepatorenal disease mechanisms.',
            'Brain mapping hypothesis development in neurological research and cognitive function.',
            'Complex glycemic control in five male Japanese patients with neurological and hepatorenal conditions.',
            'Genetic heterogeneity in breast cancer research and oncological treatment approaches.',
            'Defibrillator objective testing in cardiovascular emergency medicine and patient care.',
            'Hyperosmolar conditions in 45-year-old patients with hepatorenal complications.',
            'Pioneering research questions in hepatorenal syndrome and treatment innovations.',
            'Convulsive background analysis in neurological and oncological patient populations.',
            'Granulosa cell background research in hepatorenal and oncological disease mechanisms.',
            'Behavioral analysis of Garcinellipt effects on neurological function and cognition.',
            'Epilepsy study design for cardiovascular complications and neurological outcomes.',
            'Anti-oxidant effects of Dexamethasone in neurological and cardiovascular systems.',
            'Cardiac glycoside research questions in neurological disorders and heart function.'
        ],
        'group': [
            'neurological',
            'hepatorenal',
            'cardiovascular|hepatorenal',
            'neurological',
            'neurological',
            'neurological',
            'cardiovascular|hepatorenal',
            'cardiovascular',
            'neurological',
            'neurological',
            'neurological|hepatorenal|oncological',
            'oncological',
            'neurological',
            'neurological|hepatorenal',
            'neurological|oncological',
            'neurological|hepatorenal',
            'neurological',
            'neurological|hepatorenal',
            'oncological',
            'cardiovascular',
            'hepatorenal',
            'hepatorenal',
            'neurological|oncological',
            'hepatorenal|oncological',
            'neurological',
            'cardiovascular',
            'neurological|cardiovascular',
            'neurological'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Guardar dataset
    df.to_csv('data/medical_literature_dataset.csv', index=False)
    print(f"Dataset creado con {len(df)} registros")
    print(f"Grupos únicos: {df['group'].unique()}")
    
    # Crear split de entrenamiento y prueba
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, 
                                        stratify=df['group'].str.split('|').str[0])
    
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f"Datos de entrenamiento: {len(train_df)} registros")
    print(f"Datos de prueba: {len(test_df)} registros")
    
    return df

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    create_sample_dataset()
