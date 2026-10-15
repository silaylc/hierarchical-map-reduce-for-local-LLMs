# Sistem Ozellikleri Notu

## Genel Bakis

Bu belge, DocSummarizer sisteminin su anki ozellik setini ozetlemektedir.

## Desteklenen Formatlar

Sistem su an dort farkli dosya formatini desteklemektedir: PDF, DOCX,
TXT ve MD. Her format icin ayri bir okuyucu fonksiyonu tanimlanmistir.

## Ozetleme Stratejisi

Ozetleme, Map-Reduce yaklasimiyla yapilmaktadir. Once belge chunk'lara
bolunur, her chunk icin bagimsiz bir ozet uretilir, ardindan bu ozetler
hiyerarsik olarak birlestirilir. Son asamada, uslup ve format
tutarliligi icin bir normalizasyon adimi calistirilir.

## Bilinen Sinirlamalar

Su anda sistemde iki bilinen sinirlama bulunmaktadir. Birincisi,
sifreli PDF dosyalarinin okunamamasidir. Ikincisi, cok sayida buyuk
belgenin ayni anda islenmesi durumunda HTTP timeout riskinin
bulunmasidir.
