from flask import Flask, render_template
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

data = pd.read_csv('global_unemployment_data.csv')
data.columns = data.columns.str.strip()
years = [str(y) for y in range(2014, 2025) if str(y) in data.columns]

@app.route('/')
def dashboard():
    title = "Global Unemployment Dashboard"
    team_names = [
        "Bagus Putra Wiratama, 23.11.5560",
        "Wahyu Adi Nugroho, 23.11.5533",
        "Puantorian Antasena Handoko, 23.11.5553",
        "Muhammad Fachri Agus M, 23.11.5544"
    ]

    # 1. Rata-rata pengangguran global per tahun
    global_avg = pd.DataFrame({
        'Year': [int(y) for y in years],
        'Unemployment Rate': [data[y].mean() for y in years]
    })
    avg_unemployment = global_avg['Unemployment Rate'].mean()
    max_year = global_avg.loc[global_avg['Unemployment Rate'].idxmax()]['Year']
    min_year = global_avg.loc[global_avg['Unemployment Rate'].idxmin()]['Year']

    # Prediksi 2025 global
    X = np.asarray(global_avg['Year']).reshape(-1, 1)
    y = np.asarray(global_avg['Unemployment Rate'])
    model = LinearRegression()
    model.fit(X, y)
    pred_2025_global = model.predict(np.array([[2025]]))[0]
    pred_df_global = pd.DataFrame({'Year': [2025], 'Unemployment Rate': [pred_2025_global]})

    # Prediksi 2025 per gender
    # Transform ke long format
    df_long = data.melt(
        id_vars=['country_name', 'sex', 'age_group', 'age_categories'],
        value_vars=years,
        var_name='year',
        value_name='unemployment_rate'
    )
    df_long['year'] = df_long['year'].astype(int)
    gender_avg = df_long.groupby(['year', 'sex'])['unemployment_rate'].mean().reset_index()
    pred_gender = []
    for gender in gender_avg['sex'].unique():
        sub = gender_avg[gender_avg['sex'] == gender]
        Xg = np.asarray(sub['year']).reshape(-1, 1)
        yg = np.asarray(sub['unemployment_rate'])
        mg = LinearRegression().fit(Xg, yg)
        pred = mg.predict(np.array([[2025]]))[0]
        pred_gender.append({'Gender': gender, 'Prediksi 2025': round(pred, 2)})
    pred_df_gender = pd.DataFrame(pred_gender)

    # Prediksi 2025 per kelompok umur
    age_avg = df_long.groupby(['year', 'age_categories'])['unemployment_rate'].mean().reset_index()
    pred_age = []
    for age in age_avg['age_categories'].unique():
        sub = age_avg[age_avg['age_categories'] == age]
        Xa = np.asarray(sub['year']).reshape(-1, 1)
        ya = np.asarray(sub['unemployment_rate'])
        ma = LinearRegression().fit(Xa, ya)
        pred = ma.predict(np.array([[2025]]))[0]
        pred_age.append({'Kelompok Umur': age, 'Prediksi 2025': round(pred, 2)})
    pred_df_age = pd.DataFrame(pred_age)

    # Plot tren global + prediksi (dengan garis putus-putus ke titik prediksi)
    fig1 = go.Figure()
    # Data historis
    fig1.add_trace(go.Scatter(x=global_avg['Year'], y=global_avg['Unemployment Rate'], mode='lines+markers', name='Global'))
    # Garis putus-putus dari tahun terakhir ke prediksi
    last_year = global_avg['Year'].iloc[-1]
    last_val = global_avg['Unemployment Rate'].iloc[-1]
    fig1.add_trace(go.Scatter(
        x=[last_year, 2025],
        y=[last_val, pred_2025_global],
        mode='lines',
        name='Prediksi 2025 (Garis)',
        line=dict(dash='dash', color='red')
    ))
    # Titik prediksi
    fig1.add_trace(go.Scatter(
        x=[2025],
        y=[pred_2025_global],
        mode='markers',
        name='Prediksi 2025',
        marker=dict(color='red', size=10, symbol='diamond')
    ))
    fig1.update_layout(title='Rata-rata Global Tingkat Pengangguran per Tahun (dengan Prediksi 2025)', xaxis_title='Tahun', yaxis_title='Tingkat Pengangguran (%)')
    plot_div_global = pio.to_html(fig1, full_html=False)

    # Tampilkan nilai prediksi di bawah grafik (global, gender, umur)
    pred_html = """
    <div class='mb-3'>
      <h6>Prediksi Rata-rata Global Pengangguran (2025)</h6>
      {global_table}
    </div>
    <div class='mb-3'>
      <h6>Prediksi Rata-rata Pengangguran per Gender (2025)</h6>
      {gender_table}
    </div>
    <div class='mb-3'>
      <h6>Prediksi Rata-rata Pengangguran per Kelompok Umur (2025)</h6>
      {age_table}
    </div>
    """.format(
        global_table=pred_df_global.round(2).to_html(index=False, classes='table table-bordered table-sm', header=True, border=0),
        gender_table=pred_df_gender.to_html(index=False, classes='table table-bordered table-sm', header=True, border=0),
        age_table=pred_df_age.to_html(index=False, classes='table table-bordered table-sm', header=True, border=0)
    )

    # 2. Rata-rata pengangguran berdasarkan gender per tahun
    gender_group = data.groupby('sex')[years].mean().T
    fig2 = go.Figure()
    for gender in gender_group.columns:
        fig2.add_trace(go.Scatter(x=gender_group.index.astype(int), y=gender_group[gender], mode='lines+markers', name=gender))
    fig2.update_layout(title='Rata-rata Pengangguran Berdasarkan Gender per Tahun', xaxis_title='Tahun', yaxis_title='Tingkat Pengangguran (%)')
    plot_div_gender = pio.to_html(fig2, full_html=False)

    # 3. Rata-rata pengangguran berdasarkan kelompok umur per tahun
    age_group = data.groupby('age_categories')[years].mean().T
    fig3 = go.Figure()
    for age in age_group.columns:
        fig3.add_trace(go.Scatter(x=age_group.index.astype(int), y=age_group[age], mode='lines+markers', name=age))
    fig3.update_layout(title='Rata-rata Pengangguran Berdasarkan Kelompok Umur per Tahun', xaxis_title='Tahun', yaxis_title='Tingkat Pengangguran (%)')
    plot_div_age = pio.to_html(fig3, full_html=False)

    # 4. Top 10 negara dengan rata-rata pengangguran tertinggi (2014-2024)
    data['avg_unemp'] = data[years].mean(axis=1)
    country_avg = data.groupby('country_name')['avg_unemp'].mean()
    top_countries = country_avg.sort_values(ascending=False).head(10)  # type: ignore
    fig4 = go.Figure([go.Bar(x=top_countries.index, y=top_countries.values)])
    fig4.update_layout(title='Top 10 Negara dengan Rata-rata Pengangguran Tertinggi (2014-2024)', xaxis_title='Negara', yaxis_title='Rata-rata Pengangguran (%)')
    plot_div_top_countries = pio.to_html(fig4, full_html=False)

    return render_template(
        'dashboard.html',
        title=title,
        team_names=team_names,
        avg_unemployment=avg_unemployment,
        max_country=f"Tahun tertinggi: {max_year}",
        min_country=f"Tahun terendah: {min_year}",
        plot_div_global=plot_div_global,
        plot_div_gender=plot_div_gender,
        plot_div_age=plot_div_age,
        plot_div_top_countries=plot_div_top_countries,
        pred_df_global=pred_df_global,
        pred_df_gender=pred_df_gender,
        pred_df_age=pred_df_age
    )

if __name__ == '__main__':
    app.run(debug=True) 