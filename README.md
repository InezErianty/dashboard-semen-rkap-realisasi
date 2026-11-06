# Dashboard RKAP vs Realisasi Semen

Dashboard interaktif untuk membandingkan target (RKAP) dan aktual (Realisasi)
produksi serta rilis semen PT Semen Indonesia, per tipe semen dan per bulan,
hingga level Plant dan Finish Mill.

## Cara menjalankan

1. Pastikan Python 3.9 atau lebih baru sudah terpasang.
2. Pasang kebutuhan paket:

   pip install -r requirements.txt

3. Jalankan aplikasi dari dalam folder ini:

   streamlit run app.py

4. Aplikasi akan terbuka otomatis di browser pada alamat http://localhost:8501

## Struktur folder

semen_dashboard/
  app.py                 program utama
  requirements.txt       daftar paket
  README.md              berkas ini
  data/                  berisi file Excel sumber data
  .streamlit/config.toml tema tampilan

## Fitur

Panel kiri (filter): parameter (Produksi atau Rilis), tipe semen, dan rentang bulan.

Baris KPI: total RKAP, total Realisasi, persen pencapaian, dan selisih.

Tab Ringkasan: grafik RKAP vs Realisasi per bulan, komposisi tipe semen,
dan pencapaian per tipe semen terhadap target 100 persen.

Tab Tren Bulanan: garis tren realisasi tiap tipe semen sepanjang tahun.

Tab Plant dan Finish Mill: rincian realisasi produksi per Plant Tuban 1 sampai 4
dan per Finish Mill (FM 1 sampai 9). Hanya aktif untuk parameter Produksi.

Tab Data: tabel data terfilter yang dapat diunduh sebagai CSV.

## Mengganti data

Gunakan tombol unggah di panel kiri untuk memakai file Excel tahun lain,
selama format kolomnya sama dengan file bawaan (Source, Parameter, Plant, FM,
Tipe Semen, UoM, kolom bulan, Total).

Rumus pencapaian yang dipakai:
Pencapaian (persen) = Realisasi / RKAP * 100
