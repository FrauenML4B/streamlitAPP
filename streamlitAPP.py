import numpy as np
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk
from io import StringIO

d = None
max_files = 3
dfs = []
merged_df2 = None
uploaded_files = st.file_uploader(
    "3 Dateien des Sensor Loggers bitte hochladen in folgender Reihenfolge: Acceleratoren-, Gyrosscope- und "
    "Gravitydaten!:",
    type={"csv"},
    accept_multiple_files=True)
i = 1
counter = 0
if uploaded_files is not None:
    for up in uploaded_files:
        st.write(f'Datei Nr. {i}:', up.name)
        i = i + 1
    if len(uploaded_files) > max_files:
        st.error(f'Bitte lade nicht mehr als 3 Dateien hoch!')
    elif len(uploaded_files) < max_files:
        st.error(f'Bitte lade mindestens 3 Dateien hoch!')
    else:

        for uploaded_file in uploaded_files:
            string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(string_data)
            df1 = pd.DataFrame(df)
            df1 = df1.drop(columns=['time', 'seconds_elapsed'])
            # st.write("df1")
            # st.write(df1)
            if counter == 0:
                df1 = df1.rename(columns={'z': 'acc_z', 'y': 'acc_y', 'x': 'acc_x'})
            elif counter == 1:
                df1 = df1.rename(columns={'z': 'gravity_z', 'y': 'gravity_y', 'x': 'gravity_x'})
            elif counter == 2:
                df1 = df1.rename(columns={'z': 'gyro_z', 'y': 'gyro_y', 'x': 'gyro_x'})

            counter = counter + 1
            dfs.append(df1)

            # merged_df = pd.merge(dfs[0],dfs[1],left_index=True, right_index=True)
            # st.write(merged_df)
        pass

    if dfs:
        # st.write(dfs[0])
        # st.write(dfs[1])
        # st.write(dfs[2])
        merged_df = pd.merge(dfs[0], dfs[1], left_index=True, right_index=True)
        # st.write(merged_df)
        merged_df2 = pd.merge(merged_df, dfs[2], left_index=True, right_index=True)
        html = merged_df2.to_html()
        html = f"""<style> table{{font-size:15px; margin-left:auto; margin-right:auto;}}</style>{html}"""
        # st.write(html,unsafe_allow_html=True)
        st.write("Acceleratoren Daten:")
        st.line_chart(dfs[0])
        st.write("Gravity Daten:")
        st.line_chart(dfs[1])
        st.write("Gyrosscope Daten:")
        st.line_chart(dfs[2])
    else:
        pass
if dfs:
    data = pd.read_excel('TestDataFinal.xlsx')
    df = pd.DataFrame(data)
    shuffled_df = df.sample(frac=1)
    # st.write(df)
    np.random.seed(42)
    X = shuffled_df.drop('target', axis=1)
    y = shuffled_df["target"]
    # st.write(y)
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.5)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    pred1 = clf.predict(merged_df2)
    result = pd.DataFrame({'Vorhersage':pred1})
    merged_df3 = pd.merge(merged_df2,result,left_index=True, right_index=True)
    st.write(merged_df3)










