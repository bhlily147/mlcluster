from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pymysql
import seaborn as sns
from sklearn.cluster import KMeans


def getConnect(server, port, database, username, password):
    return pymysql.connect(
        host=server,
        port=int(port),
        database=database,
        user=username,
        password=password,
    )

def closeConnection(conn):
    if conn is not None:
        conn.close()

def queryDataset(conn, sql):
    with conn.cursor() as cursor:
        cursor.execute(sql)
        columns = [column[0] for column in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns)

def showHistogram(df, columns):
    plt.figure(figsize=(7, 8))
    for n, column in enumerate(columns, start=1):
        plt.subplot(3, 1, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.histplot(df[column], bins=32, kde=True)
        plt.title(f'Histogram of {column}')
    plt.show()

def elbowMethod(df, columnsForElbow):
    X = df.loc[:, columnsForElbow].values
    cluster_counts = np.arange(1, min(10, len(X)) + 1)
    inertia = []
    for n in cluster_counts:
        model = KMeans(
            n_clusters=n,
            init='k-means++',
            max_iter=500,
            random_state=42,
        )
        model.fit(X)
        inertia.append(model.inertia_)

    if len(cluster_counts) >= 3:
        improvements = (np.array(inertia[:-1]) - inertia[1:]) / inertia[:-1]
        drops = improvements[:-1] - improvements[1:]
        optimal_k = int(cluster_counts[1:-1][np.argmax(drops)])
    else:
        optimal_k = int(cluster_counts[-1])

    plt.figure(figsize=(15, 6))
    plt.plot(cluster_counts, inertia, 'o-')
    plt.axvline(
        optimal_k,
        color='red',
        linestyle='--',
        label=f'Optimal K = {optimal_k}',
    )
    plt.legend()
    plt.xlabel('Number of Clusters')
    plt.ylabel('Cluster sum of squared distances')
    plt.show()
    return optimal_k

def runKMeans(X, cluster):
    model = KMeans(
        n_clusters=cluster,
        init='k-means++',
        max_iter=500,
        random_state=42,
    )
    model.fit(X)
    labels = model.labels_
    return labels, model.cluster_centers_, labels

def visualizeKMeans(X, y_kmeans, cluster, title, xlabel, ylabel, colors):
    plt.figure(figsize=(10, 10))
    for i in range(cluster):
        plt.scatter(
            X[y_kmeans == i, 0],
            X[y_kmeans == i, 1],
            s=100,
            c=colors[i],
            label=f'Cluster {i + 1}',
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def export_df2excel(df, filename):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for cluster_id, cluster_df in df.groupby('cluster'):
            cluster_df.to_excel(
                writer,
                sheet_name=f'Cluster_{cluster_id}',
                index=False,
            )

def export_df2web(df, filename):
    html = """<!doctype html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Phân cụm khách hàng</title>
    <style>
        :root {
            font-family: Inter, "Segoe UI", Arial, sans-serif;
            color: #202b3c;
            background: #f3f6fb;
        }
        * { box-sizing: border-box; }
        body { margin: 0; }
        .page { max-width: 1180px; margin: 36px auto; padding: 0 24px; }
        header { margin-bottom: 24px; }
        .eyebrow { color: #5272c4; font-size: 12px; font-weight: 700; letter-spacing: .12em; }
        h1 { margin: 8px 0; font-size: 30px; }
        h2 { margin: 0; font-size: 18px; }
        .muted { color: #748094; }
        .cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 20px 0; }
        .card { padding: 20px; background: #fff; border: 1px solid #e6eaf1;
            border-radius: 16px; box-shadow: 0 10px 26px #24324a0a; }
        .metric { color: #748094; font-size: 14px; }
        .metric strong { display: block; margin-top: 8px; color: #202b3c; font-size: 27px; }
        .visuals { display: grid; grid-template-columns: .9fr 1.35fr; gap: 16px; }
        .card-title { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
        .hint { color: #8994a5; font-size: 13px; }
        .pie-wrap { max-width: 280px; margin: 20px auto 4px; }
        #legend { display: flex; justify-content: center; flex-wrap: wrap; gap: 12px; }
        .legend-item { display: flex; align-items: center; gap: 6px; color: #637087; font-size: 13px; }
        .dot { width: 9px; height: 9px; border-radius: 50%; }
        #bars { display: grid; gap: 18px; margin-top: 24px; }
        .bar-row { display: grid; grid-template-columns: 62px 1fr 42px; align-items: center;
            gap: 12px; width: 100%; padding: 3px 0; border: 0; background: transparent;
            color: #56647a; font: inherit; text-align: left; cursor: pointer; }
        .bar-row:hover .bar-track, .bar-row.active .bar-track { background: #e1e8f4; }
        .bar-track { height: 10px; overflow: hidden; border-radius: 8px; background: #edf1f6; }
        .bar-fill { height: 100%; border-radius: inherit; transition: width .35s ease; }
        .bar-count { text-align: right; font-size: 13px; }
        .table-card { margin-top: 16px; }
        .table-head { display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }
        .table-tools { display: flex; align-items: center; gap: 8px; }
        input, .reset { padding: 10px 12px; border: 1px solid #dfe5ee; border-radius: 9px;
            background: #fff; color: #344158; font: inherit; }
        .reset { cursor: pointer; }
        .reset:hover { border-color: #4969cf; color: #3e5fc2; }
        .table-wrap { max-height: 520px; overflow: auto; border: 1px solid #e5eaf1; border-radius: 11px; }
        table { width: 100%; border-collapse: collapse; font-size: 14px; }
        th, td {
            padding: 11px 13px;
            border-bottom: 1px solid #e8edf3;
            text-align: left;
            white-space: nowrap;
        }
        th { position: sticky; top: 0; color: #41516b; background: #f4f7fb; }
        tbody tr:hover { background: #f8faff; }
        .empty { padding: 28px; color: #748094; text-align: center; }
        @media (max-width: 760px) {
            .page { margin: 22px auto; padding: 0 14px; }
            .cards, .visuals { grid-template-columns: 1fr; }
            .card { padding: 16px; }
            .table-tools { width: 100%; }
            input { flex: 1; min-width: 0; }
        }
    </style>
</head>
<body>
    <main class="page">
        <header>
            <div class="eyebrow">CUSTOMER ANALYTICS</div>
            <h1>Phân cụm khách hàng</h1>
        </header>
        <section class="cards">
            <div class="card metric">Số khách hàng<strong id="customer-count">—</strong></div>
            <div class="card metric">Số cụm<strong id="cluster-count">—</strong></div>
            <div class="card metric">Số khách đang xem<strong id="visible-count">—</strong></div>
        </section>
        <section class="visuals">
            <article class="card">
                <div class="card-title">
                    <h2>Tỷ lệ theo cụm</h2>
                </div>
                <div class="pie-wrap"><canvas id="pie"></canvas></div>
                <div id="legend"></div>
            </article>
            <article class="card">
                <div class="card-title">
                    <h2>Số lượng khách hàng</h2>
                    <span class="hint">Chọn thanh để lọc</span>
                </div>
                <div id="bars"></div>
            </article>
        </section>
        <section class="card table-card">
            <div class="table-head">
                <div>
                    <h2>Danh sách khách hàng</h2>
                    <div id="filter-label" class="hint">Tất cả các cụm</div>
                </div>
                <div class="table-tools">
                    <input id="search" type="search" placeholder="Tìm khách hàng...">
                    <button id="reset" class="reset">Hiện tất cả</button>
                </div>
            </div>
            <div class="table-wrap">
                <table><thead id="table-head"></thead><tbody id="table-body"></tbody></table>
            </div>
        </section>
    </main>
    <script src="https://cdn.sheetjs.com/xlsx-0.20.3/package/dist/xlsx.full.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
    <script>
        const colors = ['#4969cf', '#21a785', '#efa640', '#9267cf'];
        const state = {groups: [], cluster: 'all', search: ''};
        let pieChart;

        function setCluster(name) {
            state.cluster = name;
            document.querySelector('#filter-label').textContent = name === 'all'
                ? 'Tất cả các cụm'
                : `Đang xem ${name.replace('Cluster_', 'Cụm ')}`;
            document.querySelectorAll('.bar-row').forEach(button => {
                button.classList.toggle('active', button.dataset.cluster === name);
            });
            renderTable();
        }

        function renderBars() {
            const total = state.groups.reduce((sum, group) => sum + group.rows.length, 0);
            const bars = state.groups.map((group, index) => {
                const percent = group.rows.length / total * 100;
                const title = `Cụm ${index + 1}: ${group.rows.length} khách (${percent.toFixed(1)}%)`;
                return `<button class="bar-row" data-cluster="${group.name}" title="${title}">
                    <span>Cụm ${index + 1}</span>
                    <span class="bar-track">
                        <span class="bar-fill"
                              style="display:block;width:${percent}%;background:${colors[index]}"></span>
                    </span>
                    <span class="bar-count">${group.rows.length}</span>
                </button>`;
            }).join('');
            document.querySelector('#bars').innerHTML = bars;
            document.querySelectorAll('.bar-row').forEach(button => {
                button.addEventListener('click', () => setCluster(button.dataset.cluster));
            });
        }

        function renderPie() {
            const total = state.groups.reduce((sum, group) => sum + group.rows.length, 0);
            pieChart?.destroy();
            pieChart = new Chart(document.querySelector('#pie'), {
                type: 'pie',
                data: {
                    labels: state.groups.map((_, index) => `Cụm ${index + 1}`),
                    datasets: [{
                        data: state.groups.map(group => group.rows.length),
                        backgroundColor: colors,
                        borderColor: '#fff',
                        borderWidth: 4,
                        hoverOffset: 12
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {display: false},
                        tooltip: {callbacks: {
                            label: item => {
                                const percent = (item.raw / total * 100).toFixed(1);
                                return `${item.label}: ${item.raw} khách (${percent}%)`;
                            }
                        }}
                    },
                    onClick: (_, items) => {
                        if (items.length) setCluster(state.groups[items[0].index].name);
                    }
                }
            });
            document.querySelector('#legend').innerHTML = state.groups.map((group, index) =>
                `<span class="legend-item">` +
                `<i class="dot" style="background:${colors[index]}"></i>` +
                `Cụm ${index + 1}</span>`
            ).join('');
        }

        function renderTable() {
            const groups = state.groups.filter(group =>
                state.cluster === 'all' || group.name === state.cluster
            );
            const query = state.search.toLocaleLowerCase();
            const rows = groups.flatMap(group => group.rows).filter(row =>
                Object.values(row).some(value => String(value).toLocaleLowerCase().includes(query))
            );
            const columns = Object.keys(state.groups[0]?.rows[0] || {});
            document.querySelector('#visible-count').textContent = rows.length;
            const headings = columns.map(column =>
                `<th>${column === 'cluster' ? 'Cụm' : column}</th>`
            ).join('');
            document.querySelector('#table-head').innerHTML = `<tr>${headings}</tr>`;
            const body = rows.map(row => {
                const cells = columns.map(column => `<td>${row[column] ?? ''}</td>`).join('');
                return `<tr>${cells}</tr>`;
            }).join('');
            document.querySelector('#table-body').innerHTML = body ||
                `<tr><td class="empty" colspan="${columns.length}">Không tìm thấy khách hàng.</td></tr>`;
        }

        async function loadWorkbook() {
            if (location.protocol === 'file:') {
                console.error('H?y m? trang qua GitHub Pages ho?c m?y ch? HTTP ?? ??c file Excel.');
                return;
            }
            try {
                const response = await fetch('./customers_cluster.xlsx');
                if (!response.ok) throw new Error('Không tìm thấy customers_cluster.xlsx.');
                const workbook = XLSX.read(await response.arrayBuffer(), {type: 'array'});
                state.groups = workbook.SheetNames.map(name => ({
                    name,
                    rows: XLSX.utils.sheet_to_json(workbook.Sheets[name], {defval: ''})
                }));
                const total = state.groups.reduce((sum, group) => sum + group.rows.length, 0);
                document.querySelector('#customer-count').textContent = total;
                document.querySelector('#cluster-count').textContent = state.groups.length;
                renderBars();
                renderPie();
                setCluster('all');
            } catch (error) {
                console.error(error);
            }
        }

        document.querySelector('#search').addEventListener('input', event => {
            state.search = event.target.value.trim();
            renderTable();
        });
        document.querySelector('#reset').addEventListener('click', () => setCluster('all'));
        loadWorkbook();
    </script>
</body>
</html>"""
    Path(filename).write_text(html, encoding='utf-8')

def visualize3DKmeans(df, columns, hover_data, cluster):
    fig = px.scatter_3d(
        df,
        x=columns[0],
        y=columns[1],
        z=columns[2],
        color='cluster',
        hover_data=hover_data,
        category_orders={'cluster': range(cluster)},
    )
    fig.update_layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    fig.show()

conn = getConnect('localhost', 3306, 'salesdatabase', 'root', '@Obama123')
try:
    sql = (
        'select distinct customer.CustomerId, Age, Annual_Income, Spending_Score '
        'from customer, customer_spend_score '
        'where customer.CustomerId=customer_spend_score.CustomerID'
    )
    df2 = queryDataset(conn, sql)
finally:
    closeConnection(conn)

df2.columns = ['CustomerId', 'Age', 'Annual Income', 'Spending Score']
showHistogram(df2, df2.columns[1:])

columns = ['Age', 'Spending Score']
elbowMethod(df2, columns)
X = df2.loc[:, columns].values
cluster = 4
colors = ['red', 'green', 'blue', 'purple', 'black', 'pink', 'orange']
y_kmeans, centroids, labels = runKMeans(X, cluster)
df2['cluster'] = labels
visualizeKMeans(
    X, y_kmeans, cluster, 'Clusters of Customers - Age X Spending Score',
    'Age', 'Spending Score', colors,
)

columns3d = ['Age', 'Annual Income', 'Spending Score']
X3d = df2.loc[:, columns3d].values
labels3d, centroids3d, _ = runKMeans(X3d, 6)
df3 = df2.copy()
df3['cluster'] = labels3d
export_df2excel(df2, Path(__file__).with_name('customers_cluster.xlsx'))
export_df2web(df2, Path(__file__).with_name('index.html'))
visualize3DKmeans(df3, columns3d, df3.columns, 6)