import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sns.set_theme(style='whitegrid')
st.set_page_config(page_title='E-Commerce Dashboard', page_icon='🛒', layout='wide')

DATA_FILE = Path(__file__).parent / 'main_data.csv'


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=[
        'order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date'])
    return df


df = load_data()

# ---------------- Sidebar ----------------
st.sidebar.header('Filter Data')
min_d = df['order_purchase_timestamp'].min().date()
max_d = df['order_purchase_timestamp'].max().date()
date_range = st.sidebar.date_input('Rentang tanggal pembelian', value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d)
states = sorted(df['customer_state'].dropna().unique())
sel_states = st.sidebar.multiselect('State pelanggan', states, default=[],
                                    help='Kosongkan untuk menampilkan semua state')
top_n = st.sidebar.slider('Jumlah kategori teratas', 5, 15, 10)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = date_range
else:
    start = end = date_range[0] if isinstance(date_range, (tuple, list)) else date_range

f = df[(df['order_purchase_timestamp'].dt.date >= start) & (df['order_purchase_timestamp'].dt.date <= end)]
if sel_states:
    f = f[f['customer_state'].isin(sel_states)]

st.title('🛒 E-Commerce Public Dataset Dashboard')
st.caption('Pesanan berstatus delivered, Januari 2017 – Agustus 2018. Revenue = jumlah harga produk (tanpa ongkir), dalam R$.')

if f.empty:
    st.warning('Tidak ada data untuk filter yang dipilih.')
    st.stop()

orders = f.drop_duplicates('order_id').copy()
orders['terlambat'] = orders['order_delivered_customer_date'] > orders['order_estimated_delivery_date']

# ---------------- KPI ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric('Total Pesanan', f"{orders['order_id'].nunique():,}")
c2.metric('Total Revenue', f"R$ {f['price'].sum():,.0f}")
c3.metric('Rata-rata Review', f"{orders['review_score'].mean():.2f}")
c4.metric('Pesanan Terlambat', f"{orders['terlambat'].mean() * 100:.1f}%")

# ---------------- Tren ----------------
st.subheader('Tren Bulanan')
monthly = (f.assign(bulan=f['order_purchase_timestamp'].dt.to_period('M').dt.to_timestamp())
           .groupby('bulan').agg(pesanan=('order_id', 'nunique'), revenue=('price', 'sum')))
t1, t2 = st.columns(2)
with t1:
    st.markdown('**Jumlah pesanan per bulan**')
    st.line_chart(monthly['pesanan'])
with t2:
    st.markdown('**Revenue per bulan (R$)**')
    st.line_chart(monthly['revenue'])

# ---------------- Pertanyaan 1 ----------------
st.subheader('Pertanyaan 1: Kategori dengan revenue tertinggi dan review-nya')
cat = (f.groupby('product_category_name_english')
       .agg(total_revenue=('price', 'sum'), jumlah_pesanan=('order_id', 'nunique'), rata_review=('review_score', 'mean'))
       .sort_values('total_revenue', ascending=False))
top = cat.head(top_n)
avg_all = orders['review_score'].mean()

fig, axes = plt.subplots(1, 2, figsize=(16, 0.55 * top_n + 2))
sns.barplot(x=top['total_revenue'] / 1e3, y=top.index,
            palette=['#1f6fb2'] + ['#9ec3e6'] * (len(top) - 1), ax=axes[0])
axes[0].set_title(f'Top {top_n} Kategori berdasarkan Revenue')
axes[0].set_xlabel('Total Revenue (ribu R$)'); axes[0].set_ylabel('')
colors = ['#d9534f' if v < avg_all else '#5cb85c' for v in top['rata_review']]
sns.barplot(x=top['rata_review'], y=top.index, palette=colors, ax=axes[1])
axes[1].axvline(avg_all, color='black', linestyle='--', label=f'Rata-rata ({avg_all:.2f})')
axes[1].set_xlim(3, 4.6)
axes[1].set_title('Rata-rata Review Score'); axes[1].set_xlabel('Review Score'); axes[1].set_ylabel('')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
plt.tight_layout()
st.pyplot(fig)
st.caption('Hijau = review di atas rata-rata, merah = di bawah rata-rata.')
with st.expander('Lihat tabel data'):
    st.dataframe(top.round(2))

# ---------------- Pertanyaan 2 ----------------
st.subheader('Pertanyaan 2: Keterlambatan pengiriman per state dan dampaknya pada review')
min_orders = st.slider('Minimal jumlah pesanan per state', 10, 500, 100, step=10)
state = (orders.groupby('customer_state')
         .agg(jumlah_pesanan=('order_id', 'nunique'), persen_terlambat=('terlambat', 'mean'),
              rata_review=('review_score', 'mean'))
         .assign(persen_terlambat=lambda d: d['persen_terlambat'] * 100))
state = state[state['jumlah_pesanan'] >= min_orders].sort_values('persen_terlambat', ascending=False)

if state.empty:
    st.info('Tidak ada state yang memenuhi minimal jumlah pesanan.')
else:
    top_late = state.head(10).sort_values('persen_terlambat')
    overall = orders['terlambat'].mean() * 100
    status_rev = (orders.assign(status=np.where(orders['terlambat'], 'Terlambat', 'Tepat waktu'))
                  .groupby('status')['review_score'].mean().reindex(['Tepat waktu', 'Terlambat']))

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [1.6, 1]})
    axes2[0].barh(top_late.index, top_late['persen_terlambat'],
                  color=['#d9534f' if v > overall else '#9ec3e6' for v in top_late['persen_terlambat']])
    axes2[0].axvline(overall, color='black', linestyle='--', label=f'Rata-rata ({overall:.1f}%)')
    for i, v in enumerate(top_late['persen_terlambat']):
        axes2[0].text(v, i, f' {v:.1f}%', va='center', fontsize=9)
    axes2[0].set_title('State dengan Persentase Keterlambatan Tertinggi')
    axes2[0].set_xlabel('% pesanan terlambat'); axes2[0].legend(loc='lower right')

    sns.barplot(x=status_rev.index, y=status_rev.values, palette=['#5cb85c', '#d9534f'], ax=axes2[1])
    for i, v in enumerate(status_rev.values):
        if not np.isnan(v):
            axes2[1].text(i, v, f'{v:.2f}', ha='center', va='bottom')
    axes2[1].set_ylim(0, 5); axes2[1].set_xlabel(''); axes2[1].set_ylabel('Rata-rata Review Score')
    axes2[1].set_title('Review: Tepat Waktu vs Terlambat')
    plt.tight_layout()
    st.pyplot(fig2)
    with st.expander('Lihat tabel data'):
        st.dataframe(state.round(2))

# ---------------- RFM ----------------
st.subheader('Analisis Lanjutan: Segmentasi Pelanggan (RFM)')
ref_date = f['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
rfm = (f.groupby('customer_unique_id')
       .agg(last_order=('order_purchase_timestamp', 'max'), frequency=('order_id', 'nunique'),
            monetary=('price', 'sum')).reset_index())
rfm['recency'] = (ref_date - rfm['last_order']).dt.days


def segment(row):
    if row['frequency'] >= 2:
        return 'Repeat Customer'
    if row['recency'] <= 90:
        return 'New (<=90 hari)'
    if row['recency'] <= 180:
        return 'Active (91-180 hari)'
    if row['recency'] <= 365:
        return 'At Risk (181-365 hari)'
    return 'Lost (>365 hari)'


rfm['segment'] = rfm.apply(segment, axis=1)
order_seg = ['New (<=90 hari)', 'Active (91-180 hari)', 'At Risk (181-365 hari)', 'Lost (>365 hari)', 'Repeat Customer']
seg = (rfm.groupby('segment').agg(jumlah_pelanggan=('customer_unique_id', 'count'), rata_monetary=('monetary', 'mean'))
       .reindex(order_seg).dropna())

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 4.5))
sns.barplot(x=seg.index, y=seg['jumlah_pelanggan'], palette='Blues_r', ax=axes3[0])
axes3[0].set_title('Jumlah Pelanggan per Segmen'); axes3[0].set_xlabel(''); axes3[0].set_ylabel('')
sns.barplot(x=seg.index, y=seg['rata_monetary'], palette='Greens_r', ax=axes3[1])
axes3[1].set_title('Rata-rata Belanja per Segmen (R$)'); axes3[1].set_xlabel(''); axes3[1].set_ylabel('')
for ax in axes3:
    ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
st.pyplot(fig3)
st.caption('Segmentasi mengikuti rentang hari sejak pembelian terakhir terhadap tanggal terakhir pada data terfilter.')

# ---------------- Kesimpulan ----------------
st.subheader('Kesimpulan & Rekomendasi')
st.markdown("""
- **Kategori:** revenue terkonsentrasi di beberapa kategori (`health_beauty`, `watches_gifts`, `bed_bath_table`), tetapi tidak semua punya kepuasan pelanggan yang baik. Prioritaskan promosi pada kategori ber-review tinggi dan lakukan audit kualitas pada kategori ber-review rendah.
- **Pengiriman:** pesanan terlambat mendapat review jauh lebih rendah. Wilayah timur laut Brasil (AL, MA, PI, CE, SE) paling sering terlambat, sehingga perlu perbaikan estimasi dan logistik.
- **Pelanggan:** mayoritas hanya membeli sekali, sehingga kampanye *retargeting* untuk segmen At Risk/Lost dan program loyalitas layak dijalankan.
""")
