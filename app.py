from flask import Flask, render_template
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

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
    # Plot 1
    fig1 = go.Figure([go.Scatter(x=global_avg['Year'], y=global_avg['Unemployment Rate'], mode='lines+markers', name='Global')])
    fig1.update_layout(title='Rata-rata Global Tingkat Pengangguran per Tahun', xaxis_title='Tahun', yaxis_title='Tingkat Pengangguran (%)')
    plot_div_global = pio.to_html(fig1, full_html=False)

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

    return render_template('dashboard.html',
        title=title,
        team_names=team_names,
        avg_unemployment=avg_unemployment,
        max_country=f"Tahun tertinggi: {max_year}",
        min_country=f"Tahun terendah: {min_year}",
        plot_div_global=plot_div_global,
        plot_div_gender=plot_div_gender,
        plot_div_age=plot_div_age,
        plot_div_top_countries=plot_div_top_countries
    )

if __name__ == '__main__':
    app.run(debug=True) 