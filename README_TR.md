# Yerel RAG Soru-Cevap Asistanı (Foundry Local) — Türkçe Kılavuz

Bu proje, "One-Month Project Plan: Local RAG AI Assistant with Microsoft
Foundry Local" ödevinin **çalışan kodudur**. İnternete ihtiyaç duymadan,
kendi bilgisayarınızda çalışan; kendi dokümanlarınızdan (`documents/`
klasörü) yararlanarak soru cevaplayan bir yapay zeka asistanıdır.

## Proje Nasıl Çalışıyor? (Mimari)

```
Kullanıcı sorusu
     │
     ▼
1) Soru, embedding modeliyle sayısal vektöre çevrilir
     │
     ▼
2) SQLite veritabanındaki (knowledge_base.db) tüm doküman parçalarıyla
   "cosine similarity" (kosinüs benzerliği) karşılaştırılır
     │
     ▼
3) En alakalı 3 parça seçilir ("retrieval")
     │
     ▼
4) Bu parçalar, sohbet modeline "sadece bunu kullan" talimatıyla
   birlikte gönderilir ("augmented generation")
     │
     ▼
5) Model, sadece verilen bağlamı kullanarak cevap üretir ve kaynağını
   belirtir
```

## Dosyalar Ne İşe Yarıyor?

| Dosya | Görevi | Ödevdeki karşılığı |
|---|---|---|
| `documents/*.txt` | Bilgi kaynağınız (istediğiniz konuyla değiştirebilirsiniz) | Knowledge base |
| `db.py` | SQLite'a yazma/okuma fonksiyonları | Hafta 2-3: SQLite |
| `ingest.py` | Dokümanları parçalayıp embedding üretip veritabanına yazar | Hafta 3: Data Ingestion |
| `search.py` | Cosine similarity ile en alakalı parçaları bulur | Hafta 3: Retrieval |
| `main.py` | Soru-cevap döngüsü, LLM entegrasyonu | Hafta 4: LLM Integration |

---

## ADIM 1 — Foundry Local'i Bilgisayarınıza Kurun

Bu adım **Python paketinden farklıdır** — önce Foundry Local'in kendi
çalışma zamanını (runtime) kurmanız gerekir.

**Windows:**
GitHub Releases sayfasından Foundry Local installer'ını indirip kurun:
`https://github.com/microsoft/Foundry-Local/releases`

**macOS:**
```bash
brew install foundrylocal
```

Kurulumdan sonra terminalde şunu çalıştırarak servisi başlatabilirsiniz:
```bash
foundry service start
```

## ADIM 2 — Python Ortamını Hazırlayın

Python 3.11 veya üzeri gereklidir. Proje klasöründe:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Windows kullanıyorsanız:
pip install foundry-local-sdk-winml openai

# macOS / Linux kullanıyorsanız:
pip install foundry-local-sdk openai
```

## ADIM 3 — Kendi Dokümanlarınızı Ekleyin (isteğe bağlı)

`documents/` klasöründe 5 örnek `.txt` dosyası hazır durumda (RAG,
Foundry Local, embedding, SQLite ve prompt mühendisliği konularında).

İsterseniz bunları silip **kendi konunuzla ilgili** `.txt` dosyaları
koyabilirsiniz (ders notları, ürün SSS'leri, vb.). Her dosyada
paragrafları **boş satırla** ayırın — sistem paragraf paragraf
parçalıyor (chunking).

## ADIM 4 — Dokümanları İşleyin (Ingest)

```bash
python ingest.py
```

Bu komut ilk çalıştırmada embedding modelini indirir (birkaç yüz MB,
internet gerekir — **sadece bu adımda**), dokümanlarınızı parçalara
böler, her parça için embedding üretir ve `knowledge_base.db` adlı
SQLite dosyasına kaydeder.

Beklenen çıktı:
```
5 dosya bulundu: [...]
Toplam 16 chunk oluşturuldu. Embedding üretiliyor...
Embedding modeli indiriliyor: 100.0%
Bitti! 16 chunk 'knowledge_base.db' içine kaydedildi.
```

## ADIM 5 — Asistanı Çalıştırın

```bash
python main.py
```

İlk çalıştırmada sohbet modeli de (Phi/Qwen küçük bir model, ~1-2 GB)
otomatik indirilir. Sonrasında **tamamen offline** çalışır.

Örnek kullanım:
```
Soru: RAG nedir?
Cevap: RAG, bir dil modelinin cevap üretirken önce ilgili bilgiyi
bir doküman kümesinden araması... (Kaynaklar: rag_nedir.txt)

Soru: quit
```

---

## Ödev Teslimine Göre Ne Yapmalısınız?

Ödev planı 6 haftalık ama sizin 2 gününüz var — önemli olan **çalışan
bir sistem + anlaşılır bir sunum**. Öncelik sırası:

1. **Bugün:** Yukarıdaki 5 adımı çalıştırıp sistemi ayağa kaldırın.
2. **Kendi konunuzu seçin:** `documents/` içindeki örnekleri, ödev
   için uygun bir konuyla değiştirin (örn. ders notlarınız, bir
   ürün kılavuzu, SSS metni).
3. **Test edin:** Hem dokümanlarda olan hem olmayan sorular sorun;
   modelin "bu bilgi elimde yok" dediğini de gösterin (ödev bunu özellikle istiyor).
4. **README/rapor yazın:** Bu dosyayı temel alıp kısa bir Türkçe/İngilizce
   rapor hazırlayın: projenin amacı, nasıl çalıştığı, karşılaşılan
   zorluklar.
5. **Sunum:** `main.py` çalışırken canlı demo yapın — biri
   dokümanlardan cevaplanan, biri de "bilmiyorum" diyen 2 örnek soru
   gösterin.

## Sık Karşılaşılan Sorunlar

- **"foundry command not found"** → Foundry Local runtime kurulu
  değil, ADIM 1'e dönün.
- **Model indirme çok yavaş / takılıyor** → İnternet bağlantınızı
  kontrol edin; model sadece ilk seferde indirilir.
- **"documents klasöründe .txt dosyası yok"** → `documents/` klasörüne
  en az bir `.txt` dosyası koyduğunuzdan emin olun.
- **`ingest.py` her çalıştığında veritabanını sıfırlıyor** → Bu
  kasıtlı: her ingest çalıştırmasında `knowledge_base.db` temizlenip
  yeniden doldurulur, böylece eski/silinmiş dosyaların kalıntısı
  kalmaz.

## Bu Kod, Ödevdeki Hangi Kaynağa Dayanıyor?

Kod, Microsoft'un resmi öğretici sayfasındaki (`learn.microsoft.com` —
"Tutorial: Build a RAG application") örnek kodun SQLite ile
genişletilmiş halidir. `main.py` ve `ingest.py` içindeki embedding/chat
model çağrıları, resmi Foundry Local Python SDK API'sini birebir
kullanır.
